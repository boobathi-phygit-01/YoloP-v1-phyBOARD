#!/usr/bin/env python3
"""YOLOP v1 (TFLite) demo - GTK3 GUI with a staged, preallocated pipeline for the i.MX8M Plus (aarch64).

Every stage runs in its own thread. Hand-offs are "newest wins" mailboxes: a slow stage drops frames instead of
adding latency, and the picture on screen never waits for the network.

    capture --> infer --> post --> ResultBox --.
       |                                       +--> render --> GTK main thread (cairo)
       '------------ newest frame ------------'        '--> record (VPU H.264 encoder)

  capture : GStreamer (VPU decode, G2D scale + colour convert) -> view-size BGR frame in a pooled buffer
  infer   : resize + BGR->RGB into a persistent letterbox canvas, cv2.LUT straight into the int8 input tensor, NPU
  post    : decode + NMS; segmentation -> two small blend tables (keep / add) at view size
  render  : out = frame * keep / 255 + add (two OpenCV passes), boxes, HUD, BGRA for cairo
  record  : VPU H.264 encoder through GStreamer (falls back to mp4v on the CPU)

Preallocated and reused every frame: frame pool, letterbox canvas + scratch, quantization LUT, input tensor, mask
work arrays, overlay tables, BGRA display buffers and cairo surfaces, record buffers.

NEON: the script has no intrinsics. It sends the per-pixel work through OpenCV calls that have NEON paths in
typical aarch64 builds (resize, cvtColor, LUT, multiply, add) instead of NumPy fancy indexing and boolean masks.
`python3 main.py --selftest` prints what your OpenCV build supports and times each call with SIMD on and off.

Dependencies
    Debian/Ubuntu/Yocto : python3-gi  python3-gi-cairo  gir1.2-gtk-3.0  python3-numpy
    pip                 : opencv-python-headless  tflite-runtime   (or ai-edge-litert instead of tflite-runtime)
    The hardware path needs OpenCV built with GStreamer and the NXP plugins (vpudec, vpuenc_h264, imxvideoconvert_g2d).

Run
    python3 main.py --selftest
    python3 main.py
    python3 main.py --weights model/yolop_320x320_intio.tflite --anchors model/yolop-320-320_anchors.npy
    python3 main.py --source road.mp4 --view-w 960 --pin 0,2,3,1

Shortcuts:  Space pause/resume | Ctrl+S snapshot | F11 fullscreen | Ctrl+Q quit | drop a video file on the window
"""
import argparse
import ctypes
import glob
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import gi
    gi.require_version('Gtk', '3.0')
    gi.require_version('Gdk', '3.0')
    from gi.repository import Gdk, GLib, Gtk, Pango
    import cairo
except (ImportError, ValueError) as _e:
    sys.exit(f'GTK3 Python bindings are missing ({_e}).\n'
             'Install: sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0')

pc = time.perf_counter
STRIDES = (8, 16, 32)
VIDEO_PATTERNS = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm', '*.m4v', '*.ts', '*.flv')
VIEW_WIDTHS = ('480', '640', '960', '1280')

# ImageNet normalization: (x/255 - mean)/std == x*SCALE + BIAS
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SCALE = (1.0 / (255.0 * STD)).astype(np.float32)
BIAS = (-MEAN / STD).astype(np.float32)

INTERP = {'area': cv2.INTER_AREA, 'linear': cv2.INTER_LINEAR}


# ==============================================================================================
# 0. SMALL SYSTEM HELPERS
# ==============================================================================================
def tune_malloc():
    """glibc hands big freed blocks back to the OS and page-faults them in again on the next allocation. The
    per-frame temporaries that cannot be preallocated (TFLite output copies, NMS arrays) would pay that each frame."""
    try:
        libc = ctypes.CDLL('libc.so.6')
        libc.mallopt(-3, 16 << 20)      # M_MMAP_THRESHOLD: keep blocks up to 16 MB on the heap
        libc.mallopt(-1, 128 << 20)     # M_TRIM_THRESHOLD: do not give memory back every frame
    except (OSError, AttributeError):
        pass


def pin_thread(core):
    """Pin the calling thread to one CPU core (Linux: pid 0 means the calling thread). None = leave it alone."""
    if core is None or not hasattr(os, 'sched_setaffinity'):
        return
    try:
        os.sched_setaffinity(0, {core})
    except OSError:
        pass


def ensure_dst(out, dst):
    """cv2 writes into `dst` in place when shape and dtype match. If it handed back a fresh array instead,
    copy it over so the preallocated buffer always holds the result."""
    if out.ctypes.data != dst.ctypes.data:
        np.copyto(dst, out)


def even(v):
    return max(2, int(v) // 2 * 2)


def parse_cores(text):
    names = ('capture', 'infer', 'post', 'render')
    cores = {}
    if text:
        for name, c in zip(names, text.split(',')):
            c = c.strip()
            if c.isdigit():
                cores[name] = int(c)
    return cores


# ==============================================================================================
# 1. THREAD PLUMBING  (mailboxes, buffer pools, shared result, UI buffers)
# ==============================================================================================
class Stats:
    """Thread-safe timing accumulators, drained by the supervisor every report interval."""

    def __init__(self):
        self._lock = threading.Lock()
        self._t = defaultdict(float)
        self._n = defaultdict(int)

    def add(self, key, dt):
        with self._lock:
            self._t[key] += dt
            self._n[key] += 1

    def drain(self):
        with self._lock:
            t, n = dict(self._t), dict(self._n)
            self._t.clear()
            self._n.clear()
        return t, n


class Mailbox:
    """Single-slot 'newest wins' channel. put() never blocks; an item that was replaced before anyone took it
    is released (returned to its pool). get() waits for an item."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None

    def put(self, item):
        with self._cv:
            old, self._item = self._item, item
            self._cv.notify_all()
        rel = getattr(old, 'release', None)
        if rel is not None:
            rel()

    def get(self, timeout=0.2):
        with self._cv:
            self._cv.wait_for(lambda: self._item is not None, timeout)
            item, self._item = self._item, None
            if item is not None:
                self._cv.notify_all()
            return item

    def wait_empty(self, timeout):
        """Block until the consumer has taken the last item (lock-step mode for files)."""
        with self._cv:
            self._cv.wait_for(lambda: self._item is None, timeout)


class Slot:
    """One reusable frame buffer. Reference counted: it returns to the pool when the last user releases it."""
    __slots__ = ('img', 'seq', '_refs', '_lock', '_pool')

    def __init__(self, pool, shape):
        self.img = np.empty(shape, np.uint8)
        self.seq = 0
        self._refs = 0
        self._lock = threading.Lock()
        self._pool = pool

    def retain(self, n=1):
        with self._lock:
            self._refs += n

    def release(self):
        with self._lock:
            self._refs -= 1
            done = self._refs <= 0
        if done:
            self._pool._free.put(self)


class FramePool:
    def __init__(self, n, shape):
        self.shape = shape
        self._free = queue.SimpleQueue()
        for _ in range(n):
            self._free.put(Slot(self, shape))

    def acquire(self, timeout=0.1):
        try:
            slot = self._free.get(timeout=timeout)
        except queue.Empty:
            return None
        slot._refs = 1                                  # the capture thread's own hold
        return slot


class Raw:
    """Network outputs on their way to the post thread."""
    __slots__ = ('seq', 'lb', 'heads', 'seg')

    def __init__(self, seq, lb, heads, seg):
        self.seq, self.lb, self.heads, self.seg = seq, lb, heads, seg

    def release(self):
        pass


Result = namedtuple('Result', 'seq boxes ov')            # boxes (n,6) in view coordinates, ov: OverlaySlot | None


class OverlaySlot:
    """Blend tables at view size:  out = frame * keep / 255 + add."""
    __slots__ = ('keep', 'add')

    def __init__(self, h, w):
        self.keep = np.full((h, w, 3), 255, np.uint8)
        self.add = np.zeros((h, w, 3), np.uint8)


class ResultBox:
    """Latest result, shared by the post thread (writer) and the render thread (single reader). The writer never
    reuses an overlay slot that is published or being drawn."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._busy = None

    def publish(self, res):
        with self._lock:
            self._latest = res

    def borrow(self):
        with self._lock:
            self._busy = self._latest
            return self._busy

    def release(self):
        with self._lock:
            self._busy = None

    def pick_free(self, slots):
        with self._lock:
            held = {id(r.ov) for r in (self._latest, self._busy) if r is not None and r.ov is not None}
            for s in slots:
                if id(s) not in held:
                    return s
        return None


class UiBuffers:
    """Preallocated BGRA frames and cairo surfaces shared with the GTK thread. The renderer never writes to a buffer
    the UI is showing or has pending (3 buffers: showing + pending + being drawn)."""

    def __init__(self, h, w, n=3):
        self.arrays = [np.zeros((h, w, 4), np.uint8) for _ in range(n)]
        self.surfaces = [cairo.ImageSurface.create_for_data(a, cairo.FORMAT_RGB24, w, h, w * 4) for a in self.arrays]
        self._lock = threading.Lock()
        self._free = list(range(n))
        self._pending = None
        self._showing = None

    def acquire(self):
        with self._lock:
            return self._free.pop() if self._free else None

    def publish(self, i):
        """True when the UI thread has to be woken (nothing was pending)."""
        with self._lock:
            old, self._pending = self._pending, i
            if old is not None:
                self._free.append(old)
            return old is None

    def take(self):                                       # UI thread
        with self._lock:
            i, self._pending = self._pending, None
            if i is None:
                return None
            if self._showing is not None:
                self._free.append(self._showing)
            self._showing = i
            return i


# ==============================================================================================
# 2. MODEL LAYER
# ==============================================================================================
def load_interpreter(path, threads, delegate_path=None, log=print):
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
            log(f'[delegate] {delegate_path} not found -> running on CPU')
        else:
            try:
                delegates.append(load_delegate(delegate_path))
                log(f'[delegate] loaded {delegate_path}')
            except Exception as e:  # noqa: BLE001
                log(f'[delegate] failed to load {delegate_path}: {e} -> running on CPU')
    if delegates:
        try:
            interp = Interpreter(model_path=path, num_threads=threads, experimental_delegates=delegates)
            interp.allocate_tensors()
            return interp, True
        except Exception as e:  # noqa: BLE001
            log(f'[delegate] could not prepare the model with the delegate: {e} -> running on CPU')
    interp = Interpreter(model_path=path, num_threads=threads)
    interp.allocate_tensors()
    return interp, False


class Letterbox:
    """Resize + pad geometry for one frame size, with every buffer preallocated. The gray padding is written once;
    only the content region is rewritten per frame."""

    def __init__(self, H, W, h, w, interp):
        r = min(H / h, W / w)
        self.r, self.fh, self.fw, self.interp = r, h, w, interp
        self.nw, self.nh = int(round(w * r)), int(round(h * r))
        self.dw, self.dh = (W - self.nw) // 2, (H - self.nh) // 2
        self.canvas = np.full((H, W, 3), 114, np.uint8)
        self.roi = self.canvas[self.dh:self.dh + self.nh, self.dw:self.dw + self.nw]
        self.same = (self.nw, self.nh) == (w, h)
        self.scratch = None if self.same else np.empty((self.nh, self.nw, 3), np.uint8)

    def fill(self, bgr):
        """BGR frame -> RGB letterbox canvas (one resize into scratch, one colour convert into the canvas ROI)."""
        src = bgr
        if not self.same:
            out = cv2.resize(bgr, (self.nw, self.nh), dst=self.scratch, interpolation=self.interp)
            ensure_dst(out, self.scratch)
            src = self.scratch
        try:
            out = cv2.cvtColor(src, cv2.COLOR_BGR2RGB, dst=self.roi)
            ensure_dst(out, self.roi)
        except cv2.error:                                  # a backend that rejects strided output arrays
            self.roi[...] = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        return self.canvas


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


class MaskBuilder:
    """Segmentation output -> blend tables, all work arrays preallocated.

    Raw (1,H,W,4) = [da0, da1, ll0, ll1], one quantization scale for all four, so the raw difference class1 - class0
    has the sign of the dequantized one. Both differences are resized together (one 2-channel float resize, bilinear:
    same result as resizing each map and taking the argmax) and thresholded at 0. Lane lines override the area."""

    def __init__(self, lb):
        self.key = (lb.fh, lb.fw)
        self.lb = lb
        fh, fw = lb.fh, lb.fw
        self.d = np.empty((lb.nh, lb.nw, 2), np.float32)
        self.r = np.empty((fh, fw, 2), np.float32)
        self.b_da, self.b_ll, self.inside, self.area, self.tmp = (np.empty((fh, fw), np.bool_) for _ in range(5))
        self.k1 = np.empty((fh, fw), np.uint8)
        self.slots = [OverlaySlot(fh, fw) for _ in range(3)]

    def build(self, raw_seg, show_drive, show_lanes, slot):
        lb = self.lb
        s = raw_seg[0, lb.dh:lb.dh + lb.nh, lb.dw:lb.dw + lb.nw]
        np.subtract(s[..., 1::2], s[..., 0::2], out=self.d, dtype=np.float32)       # [drive, lane] logit gaps
        if lb.same:
            np.copyto(self.r, self.d)
        else:
            out = cv2.resize(self.d, (lb.fw, lb.fh), dst=self.r, interpolation=cv2.INTER_LINEAR)
            ensure_dst(out, self.r)
        np.greater(self.r[..., 0], 0, out=self.b_da)
        np.greater(self.r[..., 1], 0, out=self.b_ll)
        if not show_drive:
            self.b_da.fill(False)
        if not show_lanes:
            self.b_ll.fill(False)
        np.logical_or(self.b_da, self.b_ll, out=self.inside)
        np.logical_not(self.b_ll, out=self.tmp)
        np.logical_and(self.b_da, self.tmp, out=self.area)
        c127 = np.uint8(127)
        np.multiply(self.area, c127, out=slot.add[..., 1])                          # green (BGR) where drivable
        np.multiply(self.b_ll, c127, out=slot.add[..., 2])                          # red where lane line
        np.multiply(self.inside, c127, out=self.k1)
        np.subtract(np.uint8(255), self.k1, out=self.k1)                            # 255 outside, 128 inside
        np.copyto(slot.keep, self.k1[..., None])


class YolopEngine:
    """Model + preallocated pre/post-processing. preprocess() fills the input tensor in place."""

    def __init__(self, weights, anchors_path, threads, delegate_path=None, interp_flag=cv2.INTER_AREA, log=print):
        self.interp, self.delegate_on = load_interpreter(weights, threads, delegate_path, log)
        self.inp = self.interp.get_input_details()[0]
        _, self.H, self.W, C = [int(v) for v in self.inp['shape']]
        outs = self.interp.get_output_details()
        # Identify outputs by shape (TFLite output order is not stable)
        self.heads = sorted([o for o in outs if int(o['shape'][-1]) == 18], key=lambda o: -int(o['shape'][1]))
        segs = [o for o in outs if int(o['shape'][-1]) == 4]
        if len(self.heads) != 3 or len(segs) != 1:
            raise RuntimeError('Unexpected model outputs: ' + ', '.join(str(list(o['shape'])) for o in outs)
                               + '. Expected 3 head maps (...x18) and 1 segmentation map (...x4): use the model '
                                 'made by quantize_yolop_tflite.py from the CUT model (yolop_cut_head.py).')
        self.seg_d = segs[0]
        self.in_idx = self.inp['index']
        self.head_idx = [d['index'] for d in self.heads]
        self.seg_idx = self.seg_d['index']
        self.anchors = np.load(anchors_path).astype(np.float32)       # (3 levels, 3 anchors, 2) in pixels
        self.interp_flag = interp_flag
        self._geo = {}

        dtype = self.inp['dtype']
        scale, zp = self.inp['quantization']
        self.tensor = np.zeros([1, self.H, self.W, C], dtype)         # the only input buffer, reused every frame
        self.int_in = dtype in (np.uint8, np.int8) and bool(scale)
        if self.int_in:
            lo, hi = np.iinfo(dtype).min, np.iinfo(dtype).max
            pix = np.arange(256, dtype=np.float32)
            lut = np.stack([np.clip(np.rint((pix * SCALE[c] + BIAS[c]) / scale + zp), lo, hi)
                            for c in range(3)]).astype(dtype)         # (3, 256) for R, G, B
            # cv2.LUT with a 3-channel table = one SIMD pass for all channels. int8 results are stored as uint8 bits.
            self.lut3 = np.ascontiguousarray(lut.view(np.uint8).T.reshape(256, 1, 3))
            self.in_u8 = self.tensor[0].view(np.uint8)
        else:
            self.f32 = self.tensor[0]

        # Warm-up (the NPU compiles the graph on the first call)
        self.interp.set_tensor(self.in_idx, self.tensor)
        self.interp.invoke()

    @property
    def description(self):
        return f"{self.W}x{self.H} {np.dtype(self.inp['dtype']).name}"

    def letterbox(self, h, w):
        lb = self._geo.get((h, w))
        if lb is None:
            lb = self._geo[(h, w)] = Letterbox(self.H, self.W, h, w, self.interp_flag)
        return lb

    def preprocess(self, lb, frame):
        canvas = lb.fill(frame)
        if self.int_in:
            out = cv2.LUT(canvas, self.lut3, dst=self.in_u8)
            ensure_dst(out, self.in_u8)
        else:                                                         # float32 model
            np.multiply(canvas, SCALE, out=self.f32)
            np.add(self.f32, BIAS, out=self.f32)

    def run(self, want_masks):
        self.interp.set_tensor(self.in_idx, self.tensor)
        self.interp.invoke()                                          # the GIL is released inside
        heads = [self.interp.get_tensor(i) for i in self.head_idx]
        seg = self.interp.get_tensor(self.seg_idx) if want_masks else None
        return heads, seg

    def decode(self, heads, lb, conf_thres, iou_thres):
        rows = np.concatenate([decode_head(rh, d, s, a, conf_thres)
                               for rh, d, s, a in zip(heads, self.heads, STRIDES, self.anchors)], 0)
        boxes = non_max_suppression(rows, conf_thres, iou_thres)
        if len(boxes):
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - lb.dw) / lb.r).clip(0, lb.fw)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - lb.dh) / lb.r).clip(0, lb.fh)
        return boxes


# ==============================================================================================
# 3. SOURCE LAYER  (webcam / video file / network stream; hardware pipelines first)
# ==============================================================================================
def hw_tail(vw=None, vh=None):
    """NXP G2D (GPU 2D engine) scale + colour convert, then a cheap BGRA->BGR for OpenCV."""
    size = f',width={vw},height={vh}' if vw else ''
    return (f'imxvideoconvert_g2d ! video/x-raw,format=BGRA{size} ! '
            'videoconvert ! video/x-raw,format=BGR')


class FrameSource:
    """Opens the first pipeline that delivers a frame. read_into() copies the newest frame into a pooled buffer of
    `view_hw` size (resizing only when the source could not be scaled in hardware)."""

    def __init__(self, kind, target, cam_w, cam_h, view_w, cam_fps, hw, interp_flag, log=print):
        self.kind, self.fps, self.total, self.label = kind, None, 0, ''
        self.interp = interp_flag
        self.cap, self._first, self._scratch, self._flip, self._pos = None, None, None, None, 0
        probe = None
        if kind == 'video':
            p = cv2.VideoCapture(str(target))
            if not p.isOpened():
                raise RuntimeError(f'Could not open {target}')
            probe = (int(p.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280, int(p.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720)
            fps = p.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 1 else None
            total = p.get(cv2.CAP_PROP_FRAME_COUNT)
            self.total = int(total) if total and total > 0 else 0
            p.release()

        for label, spec in self._candidates(kind, target, cam_w, cam_h, view_w, cam_fps, hw, probe):
            log(f'[source] trying: {label}')
            cap = self._open(spec)
            if cap is None:
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                self.cap, self._first, self.label = cap, frame, label
                log(f'[source] using: {label}')
                break
            cap.release()
        else:
            what = f'camera {target}' if kind == 'webcam' else str(target)
            raise RuntimeError(f'Could not open {what} (no pipeline delivered a frame)')

        rh, rw = self._first.shape[:2]
        if rw > view_w:
            vw = even(view_w)
            self.view_hw = (even(round(rh * vw / rw)), vw)
        else:
            self.view_hw = (rh, rw)

    @staticmethod
    def _open(spec):
        try:
            cap = spec() if callable(spec) else cv2.VideoCapture(spec, cv2.CAP_GSTREAMER)
        except Exception:  # noqa: BLE001
            return None
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
        return None

    @staticmethod
    def _v4l2(index, w, h):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @staticmethod
    def _candidates(kind, target, cam_w, cam_h, view_w, fps, hw, probe):
        """(label, pipeline string | callable) in the order they are tried."""
        c = []
        if kind == 'webcam':
            W, H = cam_w or 1280, cam_h or 720
            vw = even(min(view_w, W))
            vh = even(round(H * vw / W))
            dev = f'/dev/video{int(target)}'
            live = 'appsink drop=true max-buffers=1 sync=false'
            jpg = f'v4l2src device={dev} ! image/jpeg,width={W},height={H},framerate={fps}/1'
            if hw:
                c.append(('MJPG > VPU decode > G2D scale', f'{jpg} ! vpudec ! {hw_tail(vw, vh)} ! {live}'))
                c.append(('MJPG > jpegdec > G2D scale', f'{jpg} ! jpegdec ! {hw_tail(vw, vh)} ! {live}'))
                c.append(('YUYV > G2D scale', f'v4l2src device={dev} ! '
                          f'video/x-raw,format=YUY2,width={W},height={H},framerate={fps}/1 ! '
                          f'{hw_tail(vw, vh)} ! {live}'))
            c.append(('MJPG > jpegdec > CPU scale', f'{jpg} ! jpegdec ! videoscale ! '
                      f'video/x-raw,width={vw},height={vh} ! videoconvert ! video/x-raw,format=BGR ! {live}'))
            c.append(('MJPG > jpegdec (full size)', f'{jpg} ! jpegdec ! videoconvert ! '
                      f'video/x-raw,format=BGR ! {live}'))
            c.append(('V4L2 (OpenCV)', lambda: FrameSource._v4l2(int(target), W, H)))
        elif kind == 'video':
            w0, h0 = probe
            vw = even(min(view_w, w0))
            vh = even(round(h0 * vw / w0))
            loc = str(target).replace('"', '\\"')
            if hw:
                c.append(('file > decodebin (VPU) > G2D scale',
                          f'filesrc location="{loc}" ! decodebin ! {hw_tail(vw, vh)} ! '
                          'appsink drop=false max-buffers=2 sync=false'))
            c.append(('file > OpenCV', lambda: cv2.VideoCapture(str(target))))
        else:
            uri = str(target).replace('"', '\\"')
            live = 'appsink drop=true max-buffers=1 sync=false'
            if hw and str(target).startswith('rtsp'):
                for name in ('h264', 'h265'):
                    c.append((f'RTSP {name.upper()} > VPU decode > G2D',
                              f'rtspsrc location="{uri}" latency=100 ! rtp{name}depay ! {name}parse ! vpudec ! '
                              f'{hw_tail()} ! {live}'))
            c.append(('stream > OpenCV', lambda: cv2.VideoCapture(str(target))))
        return c

    @property
    def seekable(self):
        return self.kind == 'video' and self.total > 0

    def seek(self, frac):
        if self.seekable:
            n = int(max(0.0, min(1.0, frac)) * (self.total - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            self._pos = n

    def position(self):
        return self._pos

    def read_into(self, dst, mirror=False):
        """Newest frame -> dst (view-size BGR). False at end of file / dead stream."""
        if self._first is not None:
            img, self._first = self._first, None
        else:
            ok, img = self.cap.read() if self._scratch is None else self.cap.read(self._scratch)
            if not ok or img is None:
                return False
        if self._scratch is None or img.ctypes.data != self._scratch.ctypes.data:
            self._scratch = img                               # keep it as the reusable read buffer
        if mirror:
            if self._flip is None or self._flip.shape != img.shape:
                self._flip = np.empty_like(img)
            out = cv2.flip(img, 1, dst=self._flip)
            ensure_dst(out, self._flip)
            img = self._flip
        if img.shape[:2] == dst.shape[:2]:
            np.copyto(dst, img)
        else:
            out = cv2.resize(img, (dst.shape[1], dst.shape[0]), dst=dst, interpolation=self.interp)
            ensure_dst(out, dst)
        self._pos += 1
        return True

    def release(self):
        if self.cap is not None:
            self.cap.release()


def open_writer(path, fps, w, h, use_vpu, log):
    """VPU H.264 encoder through GStreamer if possible, else software mp4v. Returns (writer, label) or (None, '')."""
    if use_vpu:
        pipe = ('appsrc ! videoconvert ! video/x-raw,format=NV12 ! vpuenc_h264 ! h264parse ! '
                f'mp4mux ! filesink location="{path}"')
        wr = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, float(fps), (w, h), True)
        if wr.isOpened():
            log('[record] VPU H.264 encoder (vpuenc_h264)')
            return wr, 'vpuenc_h264'
        wr.release()
        log('[record] VPU pipeline unavailable -> software mp4v')
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (w, h))
    if wr.isOpened():
        return wr, 'mp4v'
    wr.release()
    return None, ''


# ==============================================================================================
# 4. PIPELINE THREADS
# ==============================================================================================
@dataclass
class RunConfig:
    kind: str
    target: object
    cam_w: Optional[int]
    cam_h: Optional[int]
    weights: str
    anchors: str
    threads: int
    delegate: Optional[str]
    out_dir: str
    hw: bool = True
    view_w: int = 640
    cam_fps: int = 30
    vpu_record: bool = True
    interp: int = cv2.INTER_AREA
    cores: dict = field(default_factory=dict)


class LiveSettings:
    """Values the user can change while running; the stages just read them every frame."""

    def __init__(self):
        self.conf = 0.25
        self.iou = 0.45
        self.skip = 0
        self.show_boxes = True
        self.show_drive = True
        self.show_lanes = True
        self.show_hud = True
        self.mirror = False
        self.loop = True
        self.realtime = True
        self.paused = False
        self.record = False


def ui_call(fn, *args):
    def _run():
        fn(*args)
        return GLib.SOURCE_REMOVE
    GLib.idle_add(_run)


class Stage(threading.Thread):
    """Base for the pipeline threads: optional core pinning, and any exception stops the whole pipeline cleanly."""

    def __init__(self, pl, name):
        super().__init__(daemon=True, name=name)
        self.pl = pl

    def run(self):
        pin_thread(self.pl.cfg.cores.get(self.name))
        try:
            self.loop()
        except Exception as e:  # noqa: BLE001
            self.pl.fail(self.name, e)

    def loop(self):
        raise NotImplementedError


class CaptureThread(Stage):
    def __init__(self, pl):
        super().__init__(pl, 'capture')

    def loop(self):
        pl, s, src = self.pl, self.pl.s, self.pl.src
        seq, just_looped, t_next = 0, False, pc()
        paced = src.kind == 'video'
        while not pl.stop_evt.is_set():
            if s.paused and not pl.step_evt.is_set():
                pl.stop_evt.wait(0.03)
                t_next = pc()
                continue
            pl.step_evt.clear()
            seek, pl.seek_req = pl.seek_req, None
            if seek is not None:
                src.seek(seek)
                t_next = pc()
            slot = pl.pool.acquire(0.1)
            if slot is None:
                continue
            t0 = pc()
            ok = src.read_into(slot.img, s.mirror)
            pl.stats.add('capture', pc() - t0)
            if not ok:                                             # end of file / dead stream
                slot.release()
                if src.kind == 'video':
                    if s.loop and src.seekable and not just_looped:
                        src.seek(0.0)
                        just_looped = True
                        continue
                    pl.end('Finished')
                else:
                    pl.end('Stream ended or camera disconnected')
                return
            just_looped = False
            seq += 1
            slot.seq = seq
            slot.retain(2)                                         # one hold for infer, one for render
            pl.q_infer.put(slot)
            pl.q_render.put(slot)
            slot.release()
            if paced:
                if s.realtime and src.fps:                         # play files at their native speed
                    t_next += 1.0 / src.fps
                    delay = t_next - pc()
                    if delay > 0:
                        pl.stop_evt.wait(delay)
                    elif delay < -0.25:
                        t_next = pc()
                elif not s.realtime:                               # lock-step: no drops, as fast as the model goes
                    pl.q_infer.wait_empty(1.0)


class InferThread(Stage):
    def __init__(self, pl):
        super().__init__(pl, 'infer')

    def loop(self):
        pl, s, eng = self.pl, self.pl.s, self.pl.eng
        n = 0
        while not pl.stop_evt.is_set():
            slot = pl.q_infer.get(0.2)
            if slot is None:
                continue
            n += 1
            if (n - 1) % (int(s.skip) + 1):                        # 'skip frames': the picture still updates
                slot.release()
                continue
            want_masks = s.show_drive or s.show_lanes
            seq = slot.seq
            try:
                h, w = slot.img.shape[:2]
                lb = eng.letterbox(h, w)
                t0 = pc()
                eng.preprocess(lb, slot.img)
                t1 = pc()
            finally:
                slot.release()                                     # the tensor holds the data; free the frame early
            heads, seg = eng.run(want_masks)
            t2 = pc()
            pl.stats.add('preprocess', t1 - t0)
            pl.stats.add('network', t2 - t1)
            pl.q_post.put(Raw(seq, lb, heads, seg))


class PostThread(Stage):
    def __init__(self, pl):
        super().__init__(pl, 'post')

    def loop(self):
        pl, s, eng = self.pl, self.pl.s, self.pl.eng
        builder = None
        while not pl.stop_evt.is_set():
            raw = pl.q_post.get(0.2)
            if raw is None:
                continue
            t0 = pc()
            boxes = eng.decode(raw.heads, raw.lb, s.conf, s.iou)
            t1 = pc()
            ov = None
            if raw.seg is not None:
                if builder is None or builder.key != (raw.lb.fh, raw.lb.fw):
                    builder = MaskBuilder(raw.lb)
                ov = pl.rbox.pick_free(builder.slots)
                if ov is not None:
                    builder.build(raw.seg, s.show_drive, s.show_lanes, ov)
                pl.stats.add('masks', pc() - t1)
            pl.stats.add('decode+nms', t1 - t0)
            pl.last_ndet = len(boxes)
            pl.rbox.publish(Result(raw.seq, boxes, ov))


class RenderThread(Stage):
    def __init__(self, pl):
        super().__init__(pl, 'render')

    def loop(self):
        pl, s = self.pl, self.pl.s
        comp = None
        fps, t_prev = 0.0, pc()
        while not pl.stop_evt.is_set():
            slot = pl.q_render.get(0.2)
            if slot is None:
                continue
            idx = pl.screen.acquire()
            if idx is None:
                slot.release()
                continue
            t0 = pc()
            res = pl.rbox.borrow()
            try:
                img = slot.img
                if comp is None or comp.shape != img.shape:
                    comp = np.empty_like(img)
                ov = res.ov if res is not None else None
                if ov is not None and ov.keep.shape == img.shape:
                    out = cv2.multiply(img, ov.keep, dst=comp, scale=1.0 / 255.0)
                    ensure_dst(out, comp)
                    out = cv2.add(comp, ov.add, dst=comp)
                    ensure_dst(out, comp)
                else:
                    np.copyto(comp, img)                          # the shared frame itself is never drawn on
                if res is not None and s.show_boxes:
                    for x1, y1, x2, y2, conf, _ in res.boxes.tolist():
                        cv2.rectangle(comp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(comp, f'{conf:.2f}', (int(x1), max(int(y1) - 4, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                now = pc()
                inst = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                if s.show_hud:
                    cv2.putText(comp, f'FPS {fps:.1f}  NN {pl.nn_rate:.1f}/s', (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                if s.record:
                    pl.rec.offer(comp)
                bgra = pl.screen.arrays[idx]
                out = cv2.cvtColor(comp, cv2.COLOR_BGR2BGRA, dst=bgra)
                ensure_dst(out, bgra)
            finally:
                pl.rbox.release()
                slot.release()
            pl.stats.add('render', pc() - t0)
            if pl.screen.publish(idx):
                GLib.idle_add(pl.deliver)


class Recorder(threading.Thread):
    """Writes annotated frames. Buffers are preallocated; if the encoder falls behind, frames are dropped here
    rather than stalling the renderer."""
    NBUF = 4

    def __init__(self, pl, shape):
        super().__init__(daemon=True, name='record')
        self.pl = pl
        self.free, self.q = queue.SimpleQueue(), queue.SimpleQueue()
        for _ in range(self.NBUF):
            self.free.put(np.empty(shape, np.uint8))
        self.writer, self.path, self.dropped = None, None, 0

    def offer(self, frame):
        try:
            buf = self.free.get_nowait()
        except queue.Empty:
            self.dropped += 1
            return
        np.copyto(buf, frame)
        self.q.put(buf)

    def _open(self, shape):
        pl = self.pl
        out = Path(pl.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'yolop_{datetime.now():%Y%m%d_%H%M%S}.mp4'
        fps = pl.src.fps if pl.src.kind == 'video' and pl.src.fps else pl.cfg.cam_fps
        wr, label = open_writer(path, fps, shape[1], shape[0], pl.cfg.vpu_record, pl._log)
        if wr is None:
            pl.s.record = False
            pl._emit('on_error', f'Could not open a video writer for {path}')
            return
        self.writer, self.path = wr, path

    def _close(self):
        if self.writer is not None:
            self.writer.release()                              # sends EOS: the file is finalized
        self.writer, self.path = None, None

    def run(self):
        s = self.pl.s
        try:
            while True:
                try:
                    buf = self.q.get(timeout=0.1)
                except queue.Empty:
                    if self.pl.stop_evt.is_set():
                        break
                    if not s.record:
                        self._close()
                    continue
                try:
                    if self.writer is None and s.record:
                        self._open(buf.shape)
                    if self.writer is not None:
                        self.writer.write(buf)
                except Exception as e:  # noqa: BLE001
                    s.record = False
                    self._close()
                    self.pl._emit('on_error', f'Recording stopped: {e}')
                self.free.put(buf)
                if not s.record and self.q.empty():
                    self._close()
        finally:
            self._close()


class InferenceWorker(threading.Thread):
    """Supervisor: loads the model, opens the source, runs the stages, reports stats to the UI."""
    REPORT_EVERY = 0.5

    def __init__(self, cfg, settings, ui):
        super().__init__(daemon=True, name='pipeline')
        self.cfg, self.s, self.ui = cfg, settings, ui
        self.stop_evt = threading.Event()                  # tells every stage to finish
        self._ended = threading.Event()                    # supervisor wake-up: user stop, end of stream or error
        self.step_evt = threading.Event()                  # produce one frame even while paused (after a seek)
        self.seek_req = None
        self.reason = 'Stopped'
        self.stats = Stats()
        self.nn_rate, self.last_ndet = 0.0, 0
        self.eng = self.src = self.pool = self.screen = self.rbox = self.rec = None
        self.q_infer, self.q_render, self.q_post = Mailbox(), Mailbox(), Mailbox()

    # ---- called from the UI thread
    def stop(self):
        self.stop_evt.set()
        self._ended.set()

    def request_seek(self, frac):
        self.seek_req = frac
        self.step_evt.set()

    # ---- called from the stages
    def end(self, reason):
        self.reason = reason
        self._ended.set()

    def fail(self, who, exc):
        traceback.print_exc()
        self._emit('on_error', f'{who}: {type(exc).__name__}: {exc}')
        self.end('Stopped after an error')

    def deliver(self):                                     # runs on the GTK thread (via idle_add)
        i = self.screen.take()
        if i is not None:
            self.ui.on_frame(self, self.screen.surfaces[i], self.screen.arrays[i])
        return GLib.SOURCE_REMOVE

    def _emit(self, name, *args):
        ui_call(getattr(self.ui, name), self, *args)

    def _log(self, msg):
        print(msg, flush=True)
        self._emit('on_log', msg)

    # ---- thread body
    def run(self):
        cfg = self.cfg
        stages = []
        try:
            self._emit('on_state', 'loading', 'Loading model...')
            self.eng = YolopEngine(cfg.weights, cfg.anchors, cfg.threads, cfg.delegate, cfg.interp, log=self._log)
            self._emit('on_info', {'backend': 'NPU' if self.eng.delegate_on else 'CPU',
                                   'model': self.eng.description})
            if not self.stop_evt.is_set():
                self._log('Opening source...')
                self.src = FrameSource(cfg.kind, cfg.target, cfg.cam_w, cfg.cam_h, cfg.view_w, cfg.cam_fps,
                                       cfg.hw, cfg.interp, log=self._log)
                self._emit('on_info', {'source': self.src.label})
                vh, vw = self.src.view_hw
                self.pool = FramePool(5, (vh, vw, 3))
                self.screen = UiBuffers(vh, vw)
                self.rbox = ResultBox()
                self.rec = Recorder(self, (vh, vw, 3))
                stages = [CaptureThread(self), InferThread(self), PostThread(self), RenderThread(self), self.rec]
                for t in stages:
                    t.start()
                self._emit('on_state', 'running', '')
                self._report_loop()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._emit('on_error', f'{type(e).__name__}: {e}')
            self.reason = 'Stopped after an error'
        finally:
            self.stop_evt.set()
            for t in stages:                               # capture first: it owns the camera handle
                t.join(timeout=3.0)
            if self.src is not None:
                self.src.release()
            self._emit('on_state', 'stopped', self.reason)

    def _report_loop(self):
        src, last = self.src, pc()
        while not self._ended.is_set():
            self._ended.wait(self.REPORT_EVERY)
            now = pc()
            dt, last = max(now - last, 1e-6), now
            t, n = self.stats.drain()
            self.nn_rate = n.get('network', 0) / dt
            st = {'fps': n.get('render', 0) / dt, 'inf_fps': self.nn_rate, 'ndet': self.last_ndet,
                  'ms': {k: 1000.0 * t[k] / n[k] for k in t if n.get(k)},
                  'rec': str(self.rec.path) if self.rec.writer is not None else ''}
            if src.kind == 'video':
                pos, fr = src.position(), (src.fps or 30.0)
                st.update(progress=(pos / src.total) if src.total else 0.0,
                          t=pos / fr, dur=(src.total / fr) if src.total else 0.0)
            self._emit('on_stats', st)


# ==============================================================================================
# 5. SELF-TEST  (python3 main.py --selftest)
# ==============================================================================================
def selftest(args):
    print('--- CPU / OpenCV ---')
    feats = ''
    try:
        with open('/proc/cpuinfo') as f:
            feats = next((ln for ln in f if ln.startswith('Features')), '')
    except OSError:
        pass
    print('NEON (asimd) reported by the kernel:', 'yes' if ' asimd' in feats else 'not reported')
    print(f'OpenCV {cv2.__version__} | optimized: {cv2.useOptimized()} | threads: {cv2.getNumThreads()}')
    try:
        print('OpenCV CPU features:', cv2.getCPUFeaturesLine())
    except AttributeError:
        pass
    gst = [ln.strip() for ln in cv2.getBuildInformation().splitlines() if ln.strip().startswith('GStreamer')]
    print('OpenCV GStreamer:', gst[0] if gst else 'unknown')

    print('--- GStreamer elements (gst-inspect-1.0 --exists) ---')
    exe = shutil.which('gst-inspect-1.0')
    for el in ('vpudec', 'vpuenc_h264', 'imxvideoconvert_g2d', 'jpegdec', 'videoconvert', 'videoscale', 'mp4mux'):
        ok = bool(exe) and subprocess.run([exe, '--exists', el], capture_output=True).returncode == 0
        print(f'  {el:22s} {"yes" if ok else "NO"}')

    print('--- per-call cost, ms:  SIMD on / SIMD off ---')
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    canvas = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    small = np.empty((180, 320, 3), np.uint8)
    small2 = np.empty_like(small)
    lut3 = np.arange(256, dtype=np.uint8).reshape(256, 1, 1).repeat(3, axis=2)
    lut1 = np.arange(256, dtype=np.uint8)
    lut_dst = np.empty_like(canvas)
    keep = np.full(frame.shape, 200, np.uint8)
    add = np.full(frame.shape, 3, np.uint8)
    comp = np.empty_like(frame)
    bgra = np.empty((360, 640, 4), np.uint8)

    def numpy_lut():                                       # the method the original script used
        lut_dst[..., 0] = lut1[canvas[..., 2]]
        lut_dst[..., 1] = lut1[canvas[..., 1]]
        lut_dst[..., 2] = lut1[canvas[..., 0]]

    def numpy_overlay():                                   # the method the original script used
        layer = np.zeros_like(frame)
        sel = canvas[..., 0].mean() > 0
        layer[:, :320][sel & (frame[:, :320, 0] > 128)] = (0, 255, 0)
        blended = cv2.addWeighted(frame, 0.5, layer, 0.5, 0)
        np.copyto(comp, blended, where=(layer[..., 1] > 0)[..., None])

    tests = [
        ('resize 640x360 -> 320x180 AREA', lambda: cv2.resize(frame, (320, 180), dst=small, interpolation=cv2.INTER_AREA)),
        ('resize 640x360 -> 320x180 LINEAR', lambda: cv2.resize(frame, (320, 180), dst=small, interpolation=cv2.INTER_LINEAR)),
        ('cvtColor BGR->RGB 320x180', lambda: cv2.cvtColor(small, cv2.COLOR_BGR2RGB, dst=small2)),
        ('LUT 3-channel 320x320 (cv2.LUT)', lambda: cv2.LUT(canvas, lut3, dst=lut_dst)),
        ('LUT 3-channel 320x320 (NumPy gather)', numpy_lut),
        ('overlay multiply+add 640x360', lambda: (cv2.multiply(frame, keep, dst=comp, scale=1 / 255.0),
                                                  cv2.add(comp, add, dst=comp))),
        ('overlay masked copy 640x360 (old style)', numpy_overlay),
        ('BGR->BGRA 640x360', lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA, dst=bgra)),
    ]
    for name, fn in tests:
        row = []
        for opt in (True, False):
            cv2.setUseOptimized(opt)
            for _ in range(5):
                fn()
            t = pc()
            for _ in range(100):
                fn()
            row.append((pc() - t) / 100 * 1000)
        print(f'  {name:42s} {row[0]:7.3f} / {row[1]:7.3f}')
    cv2.setUseOptimized(True)
    print('Rows with a big gap between the two numbers use SIMD (NEON on aarch64). NumPy rows ignore the switch.')


# ==============================================================================================
# 6. GUI
# ==============================================================================================
CSS = b"""
.section-title { font-size: 8pt; font-weight: 700; letter-spacing: 1.5px; opacity: 0.6; }
.sidebar       { padding: 14px; }
.chip-caption  { font-size: 7pt; font-weight: 700; letter-spacing: 1px; opacity: 0.55; }
.chip-value    { font-size: 14pt; font-weight: 600; font-family: monospace; }
.stat-strip    { padding: 8px 16px; border-top: 1px solid alpha(currentColor, 0.12); }
.status-line   { padding: 4px 16px 6px 16px; font-size: 9pt; opacity: 0.75; }
.transport     { padding: 8px 12px; border-top: 1px solid alpha(currentColor, 0.12); }
"""


def fmt_time(sec):
    sec = int(max(0, sec))
    return f'{sec // 60}:{sec % 60:02d}'


def find_default(rel):
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        p = base / rel
        if p.is_file():
            return str(p)
    return None


class Chip(Gtk.Box):
    def __init__(self, caption, width=7):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.val = Gtk.Label(label='-', xalign=0)
        self.val.set_width_chars(width)
        self.val.get_style_context().add_class('chip-value')
        cap = Gtk.Label(label=caption.upper(), xalign=0)
        cap.get_style_context().add_class('chip-caption')
        self.pack_start(self.val, False, False, 0)
        self.pack_start(cap, False, False, 0)

    def set(self, text):
        self.val.set_text(text)


class MainWindow(Gtk.Window):
    def __init__(self, args):
        super().__init__(title='YOLOP Demo')
        self.args = args
        self.settings = LiveSettings()
        self.worker = None
        self.state = 'idle'
        self._surface = self._last_bgra = None
        self._info = {}
        self._user_seek_until = 0.0
        self._last_rec = ''
        self._fullscreen = False
        self._run_kind = 'webcam'

        self.set_default_size(1280, 780)
        Gtk.Settings.get_default().set_property('gtk-application-prefer-dark-theme', True)
        prov = Gtk.CssProvider()
        prov.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), prov,
                                                 Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        self._build_header()
        self._build_body()

        self.connect('delete-event', self._on_delete)
        self.connect('key-press-event', self._on_key)
        self.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
        self.drag_dest_add_uri_targets()
        self.connect('drag-data-received', self._on_drop)
        self._set_state('idle')

    # ------------------------------------------------------------------ layout
    def _build_header(self):
        hb = Gtk.HeaderBar(show_close_button=True, title='YOLOP Demo',
                           subtitle='Drivable area  |  lane lines  |  vehicle detection  -  TFLite')
        self.btn_run = Gtk.Button(label='Start')
        self.btn_run.set_size_request(96, -1)
        self.btn_run.connect('clicked', self._on_run_clicked)
        hb.pack_start(self.btn_run)
        self.set_titlebar(hb)

    def _section(self, title):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        lbl = Gtk.Label(label=title.upper(), xalign=0)
        lbl.get_style_context().add_class('section-title')
        box.pack_start(lbl, False, False, 0)
        return box

    @staticmethod
    def _row(text, widget, expand=True):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        lbl = Gtk.Label(label=text, xalign=0)
        lbl.set_size_request(86, -1)
        row.pack_start(lbl, False, False, 0)
        row.pack_start(widget, expand, expand, 0)
        return row

    def _check(self, text, attr, tooltip=None):
        btn = Gtk.CheckButton(label=text)
        btn.set_active(getattr(self.settings, attr))
        btn.connect('toggled', lambda b: setattr(self.settings, attr, b.get_active()))
        if tooltip:
            btn.set_tooltip_text(tooltip)
        return btn

    def _scale(self, lo, hi, step, attr):
        sc = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, lo, hi, step)
        sc.set_value(getattr(self.settings, attr))
        sc.set_digits(2)
        sc.set_value_pos(Gtk.PositionType.RIGHT)
        sc.connect('value-changed', lambda s: setattr(self.settings, attr, s.get_value()))
        return sc

    @staticmethod
    def _file_button(title, patterns, name, default=None):
        btn = Gtk.FileChooserButton.new(title, Gtk.FileChooserAction.OPEN)
        btn.set_width_chars(20)
        flt = Gtk.FileFilter()
        flt.set_name(name)
        for p in patterns:
            flt.add_pattern(p)
        btn.add_filter(flt)
        if default:
            btn.set_filename(default)
        return btn

    def _build_body(self):
        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.add(root)

        # ----- sidebar
        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        side.get_style_context().add_class('sidebar')

        # Source
        sec = self._section('Source')
        self.source_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.source_stack = Gtk.Stack()
        switcher = Gtk.StackSwitcher(stack=self.source_stack, homogeneous=True)
        self.source_box.pack_start(switcher, False, False, 0)

        cam_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.cam_combo = Gtk.ComboBoxText()
        devs = sorted(int(p[len('/dev/video'):]) for p in glob.glob('/dev/video*') if p[len('/dev/video'):].isdigit())
        for i in (devs or [0, 1]):
            self.cam_combo.append(str(i), f'Camera {i}' + (f'  (/dev/video{i})' if i in devs else ''))
        self.cam_combo.set_active(0)
        self.res_combo = Gtk.ComboBoxText()
        for rid, txt in (('', 'Camera default (1280 x 720)'), ('640x480', '640 x 480'), ('1280x720', '1280 x 720'),
                         ('1920x1080', '1920 x 1080')):
            self.res_combo.append(rid, txt)
        self.res_combo.set_active(0)
        cam_page.pack_start(self._row('Device', self.cam_combo), False, False, 0)
        cam_page.pack_start(self._row('Resolution', self.res_combo), False, False, 0)
        self.source_stack.add_titled(cam_page, 'webcam', 'Webcam')

        vid_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.video_btn = self._file_button('Select a video', VIDEO_PATTERNS, 'Video files')
        if self.args.source and os.path.isfile(self.args.source):
            self.video_btn.set_filename(os.path.abspath(self.args.source))
            self.source_stack.set_visible_child(vid_page)
        vid_page.pack_start(self._row('File', self.video_btn), False, False, 0)
        self.source_stack.add_titled(vid_page, 'video', 'Video file')

        url_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.url_entry = Gtk.Entry(placeholder_text='rtsp://...  or  http://...')
        url_page.pack_start(self._row('URL', self.url_entry), False, False, 0)
        self.source_stack.add_titled(url_page, 'url', 'Stream')

        self.source_box.pack_start(self.source_stack, False, False, 0)
        sec.pack_start(self.source_box, False, False, 0)
        side.pack_start(sec, False, False, 0)

        # Detection
        sec = self._section('Detection')
        sec.pack_start(self._row('Confidence', self._scale(0.05, 0.95, 0.05, 'conf')), False, False, 0)
        sec.pack_start(self._row('NMS IoU', self._scale(0.10, 0.95, 0.05, 'iou')), False, False, 0)
        skip = Gtk.SpinButton.new_with_range(0, 10, 1)
        skip.set_tooltip_text('Run the network once every N+1 frames; the picture keeps updating and reuses the '
                              'last result in between')
        skip.connect('value-changed', lambda b: setattr(self.settings, 'skip', int(b.get_value())))
        sec.pack_start(self._row('Skip frames', skip, expand=False), False, False, 0)
        side.pack_start(sec, False, False, 0)

        # Overlays
        sec = self._section('Overlays')
        sec.pack_start(self._check('Bounding boxes', 'show_boxes'), False, False, 0)
        sec.pack_start(self._check('Drivable area', 'show_drive', 'Segmentation is only computed when a mask '
                                   'overlay is on'), False, False, 0)
        sec.pack_start(self._check('Lane lines', 'show_lanes'), False, False, 0)
        sec.pack_start(self._check('FPS counter', 'show_hud'), False, False, 0)
        sec.pack_start(self._check('Mirror image', 'mirror'), False, False, 0)
        side.pack_start(sec, False, False, 0)

        # Playback
        sec = self._section('Video playback')
        sec.pack_start(self._check('Loop', 'loop'), False, False, 0)
        sec.pack_start(self._check('Play at native speed', 'realtime',
                                   'Off = every frame goes through the model, as fast as it can go'), False, False, 0)
        side.pack_start(sec, False, False, 0)

        # Model & runtime
        w0 = self.args.weights or find_default('model/yolop_320x320_intio.tflite')
        a0 = self.args.anchors or find_default('model/yolop-320-320_anchors.npy')
        self.weights_btn = self._file_button('Select .tflite model', ['*.tflite'], 'TFLite model', w0)
        self.anchors_btn = self._file_button('Select anchors .npy', ['*.npy'], 'NumPy anchors', a0)
        self.threads_spin = Gtk.SpinButton.new_with_range(1, 64, 1)
        self.threads_spin.set_value(self.args.threads)
        d0 = self.args.delegate
        self.delegate_check = Gtk.CheckButton(label='Use NPU delegate')
        self.delegate_check.set_active(not self.args.no_delegate and os.path.exists(d0))
        self.delegate_entry = Gtk.Entry(text=d0)
        self.view_combo = Gtk.ComboBoxText()
        for w in VIEW_WIDTHS:
            self.view_combo.append(w, f'{w} px')
        self.view_combo.set_active_id(str(self.args.view_w) if str(self.args.view_w) in VIEW_WIDTHS else '640')
        self.view_combo.set_tooltip_text('Frames are shrunk to this width before inference, drawing and recording. '
                                         'Smaller = faster.')
        self.hw_check = Gtk.CheckButton(label='Hardware pipeline (VPU decode, G2D scale)')
        self.hw_check.set_active(not self.args.no_hw)
        self.vpu_rec_check = Gtk.CheckButton(label='Record with the VPU H.264 encoder')
        self.vpu_rec_check.set_active(not self.args.no_vpu_record)
        mbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        mbox.set_margin_top(6)
        for row in (self._row('Model', self.weights_btn), self._row('Anchors', self.anchors_btn),
                    self._row('CPU threads', self.threads_spin, expand=False),
                    self._row('View width', self.view_combo)):
            mbox.pack_start(row, False, False, 0)
        for w in (self.delegate_check, self.delegate_entry, self.hw_check, self.vpu_rec_check):
            mbox.pack_start(w, False, False, 0)
        self.model_box = Gtk.Expander(label='Model & runtime')
        self.model_box.set_expanded(not (w0 and a0))
        self.model_box.add(mbox)
        side.pack_start(self.model_box, False, False, 0)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_size_request(340, -1)
        scroller.add(side)
        root.pack_start(scroller, False, False, 0)
        root.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 0)

        # ----- main column
        col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.pack_start(col, True, True, 0)

        self.infobar = Gtk.InfoBar()
        self.infobar.set_message_type(Gtk.MessageType.ERROR)
        self.infobar.set_show_close_button(True)
        self.infobar.connect('response', lambda b, r: b.hide())
        self.infobar_label = Gtk.Label(xalign=0)
        self.infobar_label.set_line_wrap(True)
        self.infobar_label.set_selectable(True)
        self.infobar.get_content_area().add(self.infobar_label)
        col.pack_start(self.infobar, False, False, 0)

        self.video = Gtk.DrawingArea()
        self.video.set_size_request(480, 300)
        self.video.set_hexpand(True)
        self.video.set_vexpand(True)
        self.video.connect('draw', self._on_draw)
        col.pack_start(self.video, True, True, 0)

        # transport bar
        tb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        tb.get_style_context().add_class('transport')
        self.transport = tb
        self.btn_pause = self._icon_button('media-playback-pause-symbolic', 'Pause', Gtk.ToggleButton)
        self.btn_pause.connect('toggled', self._on_pause_toggled)
        self.seek = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.001)
        self.seek.set_draw_value(False)
        self.seek.set_hexpand(True)
        self.seek.connect('change-value', self._on_seek)
        self.time_lbl = Gtk.Label(label='')
        self.time_lbl.set_width_chars(11)
        self.btn_snap = self._icon_button('camera-photo-symbolic', 'Snapshot', Gtk.Button)
        self.btn_snap.set_tooltip_text('Save the current frame (Ctrl+S)')
        self.btn_snap.connect('clicked', lambda b: self._snapshot())
        self.btn_rec = self._icon_button('media-record-symbolic', 'Record', Gtk.ToggleButton)
        self.btn_rec.set_tooltip_text('Record the annotated video to ' + self.args.output)
        self.btn_rec.connect('toggled', lambda b: setattr(self.settings, 'record', b.get_active()))
        tb.pack_start(self.btn_pause, False, False, 0)
        tb.pack_start(self.seek, True, True, 0)
        tb.pack_start(self.time_lbl, False, False, 0)
        tb.pack_end(self.btn_rec, False, False, 0)
        tb.pack_end(self.btn_snap, False, False, 0)
        col.pack_start(tb, False, False, 0)

        # stats strip (wraps onto a second row when the window is narrow)
        strip = Gtk.FlowBox()
        strip.set_selection_mode(Gtk.SelectionMode.NONE)
        strip.set_homogeneous(False)
        strip.set_column_spacing(22)
        strip.set_row_spacing(4)
        strip.set_max_children_per_line(12)
        strip.get_style_context().add_class('stat-strip')
        self.chips = {}
        for key, cap, w in (('fps', 'FPS', 6), ('inf', 'Inference/s', 6), ('capture', 'Capture', 8),
                            ('preprocess', 'Preprocess', 8), ('network', 'Network', 8),
                            ('decode+nms', 'Decode+NMS', 8), ('masks', 'Masks', 8), ('render', 'Render', 8),
                            ('ndet', 'Detections', 5), ('backend', 'Backend', 5)):
            self.chips[key] = Chip(cap, w)
            strip.add(self.chips[key])
        col.pack_start(strip, False, False, 0)

        # status line
        sl = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        sl.get_style_context().add_class('status-line')
        self.spinner = Gtk.Spinner()
        self.status = Gtk.Label(label='Idle', xalign=0)
        self.status.set_ellipsize(Pango.EllipsizeMode.END)
        self.model_lbl = Gtk.Label(label='', xalign=1)
        self.model_lbl.set_ellipsize(Pango.EllipsizeMode.START)
        sl.pack_start(self.spinner, False, False, 0)
        sl.pack_start(self.status, True, True, 0)
        sl.pack_end(self.model_lbl, False, False, 0)
        col.pack_start(sl, False, False, 0)

    @staticmethod
    def _icon_button(icon, text, cls):
        btn = cls(label=text)
        btn.set_image(Gtk.Image.new_from_icon_name(icon, Gtk.IconSize.BUTTON))
        btn.set_always_show_image(True)
        return btn

    # ------------------------------------------------------------------ drawing
    def _on_draw(self, widget, cr):
        w, h = widget.get_allocated_width(), widget.get_allocated_height()
        cr.set_source_rgb(0.055, 0.062, 0.075)
        cr.paint()
        if self._surface is not None:
            iw, ih = self._surface.get_width(), self._surface.get_height()
            sc = min(w / iw, h / ih)
            cr.translate((w - iw * sc) / 2, (h - ih * sc) / 2)
            cr.scale(sc, sc)
            cr.set_source_surface(self._surface, 0, 0)
            cr.get_source().set_filter(cairo.FILTER_BILINEAR)
            cr.rectangle(0, 0, iw, ih)
            cr.fill()
            return False
        cr.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        for text, size, y, alpha in (('YOLOP  -  TFLite demo', 26, -8, 0.85),
                                     ('Pick a webcam, video file or stream, then press Start', 14, 24, 0.5)):
            cr.set_font_size(size)
            ext = cr.text_extents(text)
            cr.set_source_rgba(1, 1, 1, alpha)
            cr.move_to((w - ext.width) / 2, h / 2 + y)
            cr.show_text(text)
        return False

    # ------------------------------------------------------------------ state handling
    def _set_state(self, state):
        self.state = state
        idle, running = state == 'idle', state == 'running'
        self.btn_run.set_label('Start' if idle else 'Stop')
        ctx = self.btn_run.get_style_context()
        ctx.remove_class('suggested-action' if not idle else 'destructive-action')
        ctx.add_class('suggested-action' if idle else 'destructive-action')
        self.btn_run.set_sensitive(state != 'stopping')
        self.source_box.set_sensitive(idle)
        self.model_box.set_sensitive(idle)
        self.transport.set_sensitive(running)
        if state == 'loading':
            self.spinner.start()
        else:
            self.spinner.stop()

    def _set_status(self, text):
        self.status.set_text(text)

    # ------------------------------------------------------------------ user actions
    def _on_run_clicked(self, _btn):
        if self.state == 'idle':
            self._start()
        elif self.worker is not None:
            self._set_state('stopping')
            self._set_status('Stopping...')
            self.worker.stop()

    def _start(self):
        self.infobar.hide()
        weights, anchors = self.weights_btn.get_filename(), self.anchors_btn.get_filename()
        if not weights or not anchors:
            self.model_box.set_expanded(True)
            return self._show_error('Select the .tflite model and the anchors .npy file (Model & runtime).')
        kind = self.source_stack.get_visible_child_name()
        if kind == 'webcam':
            target = int(self.cam_combo.get_active_id() or 0)
        elif kind == 'video':
            target = self.video_btn.get_filename()
            if not target:
                return self._show_error('Choose a video file first.')
        else:
            target = self.url_entry.get_text().strip()
            if not target:
                return self._show_error('Enter a stream URL first.')
        cam_w = cam_h = None
        res = self.res_combo.get_active_id()
        if kind == 'webcam' and res:
            cam_w, cam_h = (int(v) for v in res.split('x'))
        cfg = RunConfig(kind, target, cam_w, cam_h, weights, anchors, int(self.threads_spin.get_value()),
                        self.delegate_entry.get_text().strip() if self.delegate_check.get_active() else None,
                        self.args.output,
                        hw=self.hw_check.get_active(),
                        view_w=int(self.view_combo.get_active_id() or 640),
                        cam_fps=self.args.cam_fps,
                        vpu_record=self.vpu_rec_check.get_active(),
                        interp=INTERP[self.args.interp],
                        cores=parse_cores(self.args.pin))
        self._run_kind = kind
        self._run_label = f'camera {target}' if kind == 'webcam' else os.path.basename(str(target))
        self.settings.paused = False
        self.btn_pause.set_active(False)
        self.btn_rec.set_active(False)
        self._surface = self._last_bgra = None
        self._info = {}
        self.video.queue_draw()
        self.worker = InferenceWorker(cfg, self.settings, self)
        self._set_state('loading')
        self._set_status('Loading model...')
        self.worker.start()

    def _on_pause_toggled(self, btn):
        self.settings.paused = btn.get_active()
        btn.set_label('Resume' if btn.get_active() else 'Pause')
        btn.set_image(Gtk.Image.new_from_icon_name(
            'media-playback-start-symbolic' if btn.get_active() else 'media-playback-pause-symbolic',
            Gtk.IconSize.BUTTON))

    def _on_seek(self, _scale, _scroll, value):
        self._user_seek_until = time.monotonic() + 0.6
        if self.worker is not None:
            self.worker.request_seek(max(0.0, min(1.0, value)))
        return False

    def _snapshot(self):
        if self._last_bgra is None:
            return
        out = Path(self.args.output)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'snapshot_{datetime.now():%Y%m%d_%H%M%S}.png'
        cv2.imwrite(str(path), cv2.cvtColor(self._last_bgra, cv2.COLOR_BGRA2BGR))
        self._set_status(f'Snapshot saved -> {path}')

    def _on_key(self, _w, ev):
        key = Gdk.keyval_name(ev.keyval)
        ctrl = bool(ev.state & Gdk.ModifierType.CONTROL_MASK)
        if key == 'space' and self.state == 'running' and not isinstance(self.get_focus(), Gtk.Entry):
            self.btn_pause.set_active(not self.btn_pause.get_active())
            return True
        if ctrl and key in ('s', 'S'):
            self._snapshot()
            return True
        if ctrl and key in ('q', 'Q'):
            self.close()
            return True
        if key == 'F11' or (key == 'Escape' and self._fullscreen):
            self._fullscreen = not self._fullscreen
            self.fullscreen() if self._fullscreen else self.unfullscreen()
            return True
        return False

    def _on_drop(self, _w, _ctx, _x, _y, data, _info, _time):
        uris = data.get_uris()
        if uris and self.state == 'idle':
            try:
                path, _ = GLib.filename_from_uri(uris[0])
            except GLib.Error:
                return
            self.video_btn.set_filename(path)
            self.source_stack.set_visible_child_name('video')
            self._set_status(f'Loaded {os.path.basename(path)} - press Start')

    def _on_delete(self, *_a):
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(3.0)
        return False

    def _show_error(self, msg):
        self.infobar_label.set_text(msg)
        self.infobar.show()

    # ------------------------------------------------------------------ callbacks from the worker (UI thread)
    def on_state(self, worker, state, msg):
        if worker is not self.worker:
            return
        if state == 'loading':
            self._set_state('loading')
        elif state == 'running':
            self._set_state('running')
            live = self._run_kind != 'video'
            self.seek.set_visible(not live and worker.cfg.kind == 'video')
            self.seek.set_value(0)
            self.time_lbl.set_text('LIVE' if live else '0:00 / 0:00')
            self._set_status(f'Running - {self._run_label}')
        elif state == 'stopped':
            self._set_state('idle')
            self.worker = None
            self.btn_rec.set_active(False)
            self._last_rec = ''
            self._set_status(msg)

    def on_info(self, worker, info):
        if worker is not self.worker:
            return
        self._info.update(info)
        if 'backend' in info:
            self.chips['backend'].set(info['backend'])
        self.model_lbl.set_text('  |  '.join(self._info[k] for k in ('model', 'backend', 'source')
                                             if self._info.get(k)))

    def on_log(self, worker, msg):
        if worker is self.worker:
            self._set_status(msg)

    def on_error(self, worker, msg):
        if worker is not self.worker:
            return
        self.btn_rec.set_active(False)
        self._show_error(msg)

    def on_frame(self, worker, surface, bgra):
        if worker is not self.worker:
            return
        surface.mark_dirty()                              # the pixels changed behind cairo's back
        self._surface = surface
        self._last_bgra = bgra                            # stays valid: the renderer skips the showing buffer
        self.video.queue_draw()

    def on_stats(self, worker, st):
        if worker is not self.worker:
            return
        c, ms = self.chips, st['ms']
        c['fps'].set(f"{st['fps']:.1f}")
        c['inf'].set(f"{st['inf_fps']:.1f}")
        for k in ('capture', 'preprocess', 'network', 'decode+nms', 'masks', 'render'):
            c[k].set(f'{ms[k]:.1f} ms' if k in ms else '-')
        c['ndet'].set(str(st['ndet']))
        if 'progress' in st:
            if time.monotonic() > self._user_seek_until:
                self.seek.set_value(st['progress'])
            self.time_lbl.set_text(f"{fmt_time(st['t'])} / {fmt_time(st['dur'])}")
        if st['rec'] != self._last_rec:
            self._set_status(f"Recording -> {st['rec']}" if st['rec'] else f'Recording saved ({self._last_rec})')
            self._last_rec = st['rec']


# ==============================================================================================
# 7. ENTRY POINT
# ==============================================================================================
def make_parser():
    p = argparse.ArgumentParser(description='YOLOP TFLite demo (GTK3, threaded, VPU/G2D/NPU)')
    p.add_argument('--weights', help='.tflite model (default: model/yolop_320x320_intio.tflite if present)')
    p.add_argument('--anchors', help='anchors .npy (default: model/yolop-320-320_anchors.npy if present)')
    p.add_argument('--source', help='video file to preselect in the GUI')
    p.add_argument('--threads', type=int, default=os.cpu_count() or 4, help='TFLite CPU threads (CPU fallback only)')
    p.add_argument('--delegate', default='/usr/lib/libvx_delegate.so', help='TFLite hardware delegate (NPU)')
    p.add_argument('--no-delegate', action='store_true', help='start with the NPU delegate unticked')
    p.add_argument('--output', default='runs/gui', help='folder for recordings and snapshots')
    p.add_argument('--view-w', type=int, default=640, help='frames are shrunk to this width (480/640/960/1280)')
    p.add_argument('--cam-fps', type=int, default=30, help='frame rate requested from the camera')
    p.add_argument('--no-hw', action='store_true', help='skip the VPU / G2D GStreamer pipelines')
    p.add_argument('--no-vpu-record', action='store_true', help='record with the software mp4v encoder')
    p.add_argument('--interp', choices=sorted(INTERP), default='area',
                   help='resize filter for the model input (area = same as the original script)')
    p.add_argument('--pin', help='CPU cores for capture,infer,post,render, e.g. 0,2,3,1 (default: no pinning)')
    p.add_argument('--cv-threads', type=int, default=1,
                   help='OpenCV internal threads; 1 avoids fighting our own stage threads')
    p.add_argument('--selftest', action='store_true', help='print NEON / GStreamer capabilities and timings, then exit')
    return p


def main():
    args = make_parser().parse_args()
    cv2.setNumThreads(args.cv_threads)
    tune_malloc()
    if not args.no_hw:
        os.environ.setdefault('GST_PLUGIN_FEATURE_RANK', 'vpudec:MAX')   # let decodebin prefer the VPU decoder
    if args.selftest:
        return selftest(args)
    win = MainWindow(args)
    win.connect('destroy', Gtk.main_quit)
    win.show_all()
    win.infobar.hide()                                    # hidden until an error occurs
    win.seek.hide()                                       # only for seekable video files
    if hasattr(GLib, 'unix_signal_add'):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, Gtk.main_quit)
    Gtk.main()


if __name__ == '__main__':
    main()
