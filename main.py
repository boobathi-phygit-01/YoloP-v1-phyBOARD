"""YOLOP v1 inference with the INT8 (or float) TFLite model from quantize_yolop_tflite.py.
Webcam, video file, stream URL or single image. NPU via the VX delegate when available.

Dependencies: a TFLite runtime + numpy + opencv-python. No TensorFlow, no PyTorch.
    pip install tflite-runtime opencv-python numpy       (or: ai-edge-litert instead of tflite-runtime)

Usage:
    python3 main.py --weights weights/yolop-320-320_raw_tf/yolop_320x320_intio.tflite --source 0
    python3 main.py --source road.mp4 --save
    python3 main.py --source image.jpg
    python3 main.py --profile --no-delegate        # CPU only, per-stage timings
Press q or Esc to quit.

The model is the CUT YOLOP graph (see yolop_cut_head.py): three raw detection maps (1xhxwx18) and one
segmentation map (1xHxWx4 = drive-area 2 ch + lane-line 2 ch). The box decoding (sigmoid, grid, anchors) is done
here in NumPy, directly on the integer outputs where possible, using the anchors saved by yolop_cut_head.py.
"""
import argparse
import glob
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

pc = time.perf_counter
STRIDES = (8, 16, 32)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ImageNet normalization: (x/255 - mean)/std == x*SCALE + BIAS
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SCALE = (1.0 / (255.0 * STD)).astype(np.float32)
BIAS = (-MEAN / STD).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# TFLite interpreter (+ optional NPU delegate with CPU fallback)
# ----------------------------------------------------------------------------------------------
def load_interpreter(path, threads, delegate_path=None):
    """Returns (interpreter, delegate_active). Falls back to CPU if the delegate is missing / fails."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter, load_delegate

    delegates = []
    if delegate_path:
        if not os.path.exists(delegate_path):
            print(f'[delegate] {delegate_path} not found -> running on CPU')
        else:
            try:
                delegates.append(load_delegate(delegate_path))
                print(f'[delegate] loaded {delegate_path}')
            except Exception as e:  # noqa: BLE001
                print(f'[delegate] failed to load {delegate_path}: {e} -> running on CPU')
    if delegates:
        try:
            interp = Interpreter(model_path=path, num_threads=threads, experimental_delegates=delegates)
            interp.allocate_tensors()
            return interp, True
        except Exception as e:  # noqa: BLE001
            print(f'[delegate] could not prepare the model with the delegate: {e} -> running on CPU')
    interp = Interpreter(model_path=path, num_threads=threads)
    interp.allocate_tensors()
    return interp, False


# ----------------------------------------------------------------------------------------------
# Pre-processing: letterbox + normalize + quantize in one pass (lookup table per colour channel)
# ----------------------------------------------------------------------------------------------
class Preprocessor:
    def __init__(self, H, W, in_detail):
        self.H, self.W = H, W
        self._cache = {}
        self.dtype = in_detail['dtype']
        scale, zp = in_detail['quantization']
        self.int_in = self.dtype in (np.uint8, np.int8) and bool(scale)
        if self.int_in:
            # pixel value (0..255) -> quantized model input, for each RGB channel. Quantization is folded
            # into the table, so the per-frame work is just 3 table lookups (cheap on ARM CPUs).
            lo, hi = np.iinfo(self.dtype).min, np.iinfo(self.dtype).max
            pix = np.arange(256, dtype=np.float32)
            self.lut = np.stack([np.clip(np.rint((pix * SCALE[c] + BIAS[c]) / scale + zp), lo, hi)
                                 for c in range(3)]).astype(self.dtype)          # (3, 256) for R, G, B

    def geometry(self, h, w):
        g = self._cache.get((h, w))
        if g is None:
            r = min(self.H / h, self.W / w)
            nw, nh = int(round(w * r)), int(round(h * r))
            dw, dh = (self.W - nw) // 2, (self.H - nh) // 2
            g = self._cache[(h, w)] = (r, dw, dh, nw, nh)
        return g

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        r, dw, dh, nw, nh = self.geometry(h, w)
        if (w, h) != (nw, nh):
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.H, self.W, 3), 114, np.uint8)                      # gray padding, BGR
        canvas[dh:dh + nh, dw:dw + nw] = bgr
        if self.int_in:
            out = np.empty((1, self.H, self.W, 3), self.dtype)                    # NHWC, RGB
            out[0, ..., 0] = self.lut[0][canvas[..., 2]]
            out[0, ..., 1] = self.lut[1][canvas[..., 1]]
            out[0, ..., 2] = self.lut[2][canvas[..., 0]]
        else:                                                                     # float32 model
            x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
            x *= SCALE
            x += BIAS
            out = x[None]
        return out, (r, dw, dh, nw, nh)


# ----------------------------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------------------------
def dequant(arr, d):
    scale, zp = d['quantization']
    if d['dtype'] in (np.uint8, np.int8) and scale:
        return (arr.astype(np.float32) - zp) * scale
    return arr.astype(np.float32)


def decode_head(raw, d, stride, anchor, conf_thres):
    """One raw head map (1, ny, nx, 18) -> candidate rows [cx, cy, w, h, obj, cls].
    Objectness is thresholded on the raw integers; only surviving anchors are dequantized and decoded."""
    _, ny, nx, ch = raw.shape
    p = raw[0].reshape(ny, nx, 3, ch // 3)                        # channel layout: anchor * 6 + k
    thr = np.log(conf_thres / (1.0 - conf_thres))                 # sigmoid(obj) > conf  <=>  logit > thr
    scale, zp = d['quantization']
    if d['dtype'] in (np.uint8, np.int8) and scale:
        mask = p[..., 4] > (thr / scale + zp)
    else:
        mask = p[..., 4] > thr
    ys, xs, an = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros((0, ch // 3), np.float32)
    s = 1.0 / (1.0 + np.exp(-dequant(p[ys, xs, an], d)))          # (n, 6)
    xy = (s[:, 0:2] * 2.0 - 0.5 + np.stack((xs, ys), 1)) * stride
    wh = (s[:, 2:4] * 2.0) ** 2 * anchor[an]
    return np.concatenate((xy, wh, s[:, 4:]), 1).astype(np.float32)


def nms_numpy(boxes, scores, iou_thres):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) <= iou_thres]
    return keep


def non_max_suppression(x, conf_thres=0.25, iou_thres=0.45):
    """(n, 6) rows [cx, cy, w, h, obj, cls] -> (m, 6) [x1, y1, x2, y2, conf, cls]."""
    if x.shape[0] == 0:
        return np.zeros((0, 6), np.float32)
    x = x.copy()
    x[:, 5:] *= x[:, 4:5]                                         # conf = obj * cls
    box = np.stack((x[:, 0] - x[:, 2] / 2, x[:, 1] - x[:, 3] / 2, x[:, 0] + x[:, 2] / 2, x[:, 1] + x[:, 3] / 2), 1)
    j = x[:, 5:].argmax(1)
    conf = x[np.arange(len(x)), 5 + j]
    det = np.concatenate((box, conf[:, None], j[:, None].astype(np.float32)), 1)[conf > conf_thres]
    if det.shape[0] == 0:
        return np.zeros((0, 6), np.float32)
    keep = nms_numpy(det[:, :4] + det[:, 5:6] * 4096, det[:, 4], iou_thres)[:300]
    return det[keep]


def seg_masks(raw_seg, geom, fh, fw):
    """raw (1,H,W,4) [da0, da1, ll0, ll1] -> two uint8 masks at frame size. Compared on the raw values: all four
    channels share one scale, so raw class1 > class0 equals the dequantized argmax. The channel difference is
    resized bilinearly (same as resizing both maps, then argmax)."""
    r, dw, dh, nw, nh = geom
    s = raw_seg[0, dh:dh + nh, dw:dw + nw].astype(np.float32)
    out = []
    for a, b in ((s[..., 1], s[..., 0]), (s[..., 3], s[..., 2])):
        d = a - b
        if (nw, nh) != (fw, fh):
            d = cv2.resize(d, (fw, fh), interpolation=cv2.INTER_LINEAR)
        out.append((d > 0).astype(np.uint8))
    return out


def overlay(im0, da, ll):
    """Green driving area, red lane lines, 50% blend, only where a mask is set."""
    layer = np.zeros_like(im0)
    layer[da > 0] = (0, 255, 0)
    layer[ll > 0] = (0, 0, 255)
    blended = cv2.addWeighted(im0, 0.5, layer, 0.5, 0)
    np.copyto(im0, blended, where=((da > 0) | (ll > 0))[..., None])


def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    dirs = glob.glob(f'{path}{sep}*')
    matches = [re.search(rf'%s{sep}(\d+)' % path.stem, d) for d in dirs]
    idx = [int(m.groups()[0]) for m in matches if m]
    return f'{path}{sep}{max(idx) + 1 if idx else 2}'


# ----------------------------------------------------------------------------------------------
# Source
# ----------------------------------------------------------------------------------------------
class LoadSource:
    """Webcam index, video file, stream URL or a single image. Yields BGR frames at native size."""

    def __init__(self, source='0', cam_width=None, cam_height=None):
        self.is_image = Path(str(source)).suffix.lower() in IMG_EXTS and Path(str(source)).is_file()
        self.cap = None
        if self.is_image:
            self.frame = cv2.imread(str(source))
            if self.frame is None:
                raise RuntimeError(f'Could not read image: {source}')
            self.done = False
            return
        src = int(source) if str(source).isnumeric() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open video source: {source}')
        if cam_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        if cam_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_image:
            if self.done:
                raise StopIteration
            self.done = True
            return self.frame.copy()
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='model/yolop_320x320_intio.tflite',
                   help='.tflite model from quantize_yolop_tflite.py')
    p.add_argument('--anchors', default='model/yolop-320-320_anchors.npy', help='anchors from yolop_cut_head.py')
    p.add_argument('--source', default='0', help='webcam index, video file, stream URL or image')
    p.add_argument('--conf-thres', type=float, default=0.25)
    p.add_argument('--iou-thres', type=float, default=0.45)
    p.add_argument('--threads', type=int, default=os.cpu_count() or 4, help='TFLite CPU threads')
    p.add_argument('--delegate', default='/usr/lib/libvx_delegate.so',
                   help='TFLite hardware delegate (NPU). Used if the file exists, else CPU')
    p.add_argument('--no-delegate', action='store_true', help='ignore the delegate and run on CPU')
    p.add_argument('--skip', type=int, default=0, help='skip the network for N frames between inferences')
    p.add_argument('--no-masks', action='store_true', help='detection only')
    p.add_argument('--profile', action='store_true', help='print per-stage timings every 60 frames')
    p.add_argument('--cam-width', type=int)
    p.add_argument('--cam-height', type=int)
    p.add_argument('--mirror', action='store_true')
    p.add_argument('--no-view', action='store_true')
    p.add_argument('--save', action='store_true', help='save annotated video / image to project/name')
    p.add_argument('--project', default='runs/detect')
    p.add_argument('--name', default='exp')
    p.add_argument('--exist-ok', action='store_true')
    return p


def print_profile(T, N, frames):
    if not frames:
        return
    parts = [f'{k}: {1000 * T[k] / max(N[k], 1):.1f} ms'
             for k in ('read', 'preprocess', 'network', 'decode+nms', 'masks', 'draw+show') if N[k]]
    print(f'[{frames} frames] avg per call -> ' + ' | '.join(parts))


def detect(opt):
    interp, delegate_on = load_interpreter(opt.weights, opt.threads, None if opt.no_delegate else opt.delegate)
    inp = interp.get_input_details()[0]
    _, H, W, C = [int(v) for v in inp['shape']]
    print(f"Input {[int(v) for v in inp['shape']]} {np.dtype(inp['dtype']).name} quant={inp['quantization']} | "
          f"threads={opt.threads} | delegate={'ON (' + opt.delegate + ')' if delegate_on else 'off (CPU)'}")

    # Identify outputs by shape (TFLite output order is not stable)
    heads = sorted([o for o in interp.get_output_details() if int(o['shape'][-1]) == 18],
                   key=lambda o: -int(o['shape'][1]))                 # finest (stride 8) first
    segs = [o for o in interp.get_output_details() if int(o['shape'][-1]) == 4]
    if len(heads) != 3 or len(segs) != 1:
        raise SystemExit('Unexpected outputs: ' + ', '.join(str(list(o['shape'])) for o in interp.get_output_details())
                         + '\nExpected 3 head maps (…x18) and 1 segmentation map (…x4): use the model made by '
                           'quantize_yolop_tflite.py from the CUT model (yolop_cut_head.py).')
    seg_d = segs[0]
    anchors = np.load(opt.anchors).astype(np.float32)                 # (3 levels, 3 anchors, 2) in pixels

    pre = Preprocessor(H, W, inp)
    src = LoadSource(opt.source, opt.cam_width, opt.cam_height)

    # Warm-up (the NPU compiles the graph on the first call)
    interp.set_tensor(inp['index'], np.zeros([1, H, W, C], inp['dtype']))
    interp.invoke()

    save_dir = None
    if opt.save:
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        save_dir.mkdir(parents=True, exist_ok=True)

    T, N = defaultdict(float), defaultdict(int)

    def tick(name, t0):
        T[name] += pc() - t0
        N[name] += 1

    last_boxes, last_da, last_ll = np.zeros((0, 6), np.float32), None, None
    vid_writer, fps_smooth = None, 0.0
    frame_i, t_begin = 0, pc()

    try:
        while True:
            t_frame = pc()

            t0 = pc()
            try:
                frame = next(src)
            except StopIteration:
                break
            if opt.mirror:
                frame = np.ascontiguousarray(frame[:, ::-1])
            tick('read', t0)
            fh, fw = frame.shape[:2]

            if frame_i % (opt.skip + 1) == 0:
                t0 = pc()
                x, geom = pre(frame)
                tick('preprocess', t0)

                t0 = pc()
                interp.set_tensor(inp['index'], x)
                interp.invoke()
                raw_heads = [interp.get_tensor(d['index']) for d in heads]
                raw_seg = None if opt.no_masks else interp.get_tensor(seg_d['index'])
                tick('network', t0)

                t0 = pc()
                r, dw, dh, nw, nh = geom
                rows = np.concatenate([decode_head(rh, d, s, a, opt.conf_thres)
                                       for rh, d, s, a in zip(raw_heads, heads, STRIDES, anchors)], 0)
                boxes = non_max_suppression(rows, opt.conf_thres, opt.iou_thres)
                if len(boxes):
                    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - dw) / r).clip(0, fw)
                    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - dh) / r).clip(0, fh)
                last_boxes = boxes
                tick('decode+nms', t0)

                if raw_seg is not None:
                    t0 = pc()
                    last_da, last_ll = seg_masks(raw_seg, geom, fh, fw)
                    tick('masks', t0)

            t0 = pc()
            if last_da is not None:
                overlay(frame, last_da, last_ll)
            for x1, y1, x2, y2, conf, _ in last_boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'{conf:.2f}', (int(x1), max(int(y1) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            fps = 1.0 / max(pc() - t_frame, 1e-6)
            fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps
            if not src.is_image:
                inf_fps = N['network'] / max(pc() - t_begin, 1e-6)
                cv2.putText(frame, f'FPS: {fps_smooth:.1f}  (network: {inf_fps:.1f}/s)', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            if opt.save:
                if src.is_image:
                    out = str(save_dir / (Path(opt.source).stem + '_yolop_tflite.jpg'))
                    cv2.imwrite(out, frame)
                    print(f'Saved {out}')
                else:
                    if vid_writer is None:
                        f = src.cap.get(cv2.CAP_PROP_FPS)
                        f = f if f and f > 1 else 30
                        sp = str(save_dir / 'yolop_tflite.mp4')
                        vid_writer = cv2.VideoWriter(sp, cv2.VideoWriter_fourcc(*'mp4v'), f, (fw, fh))
                        print(f'Recording to {sp}')
                    vid_writer.write(frame)

            if not opt.no_view:
                cv2.imshow('YOLOP TFLite', frame)
                key = cv2.waitKey(0 if src.is_image else 1) & 0xFF
                if key in (ord('q'), 27) or src.is_image:
                    break
            tick('draw+show', t0)

            frame_i += 1
            if opt.profile and frame_i % 60 == 0:
                print_profile(T, N, frame_i)
    except KeyboardInterrupt:
        print('Interrupted.')
    finally:
        src.release()
        if vid_writer is not None:
            vid_writer.release()
        cv2.destroyAllWindows()

    print_profile(T, N, max(frame_i, 1))


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)
    detect(opt)
