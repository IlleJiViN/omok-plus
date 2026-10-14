"""
Benchmarking Omok++ Engines on Ryzen 9 9950X3D and RTX 5090:
1. Native C++ Shared Library (libomok_engine.so) via ctypes
2. Numba LLVM JIT Engine (FastBoard)
3. Standard Python Engine (Board)
4. Deep Neural Network GPU Inference (PolicyValueNet on RTX 5090)
"""
import os
import sys
import time
import ctypes
import numpy as np
import torch

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir) if os.path.basename(_dir) in ("tests", "train") else _dir
if _root not in sys.path:
    sys.path.insert(0, _root)

from engine.board import Board
from engine.fast_board import FastBoard
from models.policy_value_net import PolicyValueNet

print("=" * 70)
print("      OMOK++ ENGINE & GPU ACCELERATION BENCHMARK (9950X3D + 5090)      ")
print("=" * 70)

# -------------------------------------------------------------
# 1. Native C++ Shared Library (ctypes)
# -------------------------------------------------------------
so_paths = [
    os.path.join(_root, "cpp", "libomok_engine.so"),
    os.path.join(_root, "cpp", "libomok_engine.dll"),
    "cpp/libomok_engine.so"
]
so_path = next((p for p in so_paths if os.path.exists(p)), None)

c_speed = None
if so_path:
    try:
        class Coord(ctypes.Structure):
            _fields_ = [("r", ctypes.c_int8), ("c", ctypes.c_int8)]

        class CaptureRecord(ctypes.Structure):
            _fields_ = [("captured", Coord * 8), ("count", ctypes.c_int8)]

        c_lib = ctypes.CDLL(so_path)
        c_lib.create_board.restype = ctypes.c_void_p
        c_lib.free_board.argtypes = [ctypes.c_void_p]
        c_lib.reset_board.argtypes = [ctypes.c_void_p]
        c_lib.place_stone_c.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int8, ctypes.POINTER(CaptureRecord)]
        c_lib.place_stone_c.restype = ctypes.c_int
        c_lib.undo_stone_c.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int8, ctypes.POINTER(CaptureRecord)]
        c_lib.check_game_status_c.argtypes = [ctypes.c_void_p]
        c_lib.check_game_status_c.restype = ctypes.c_int

        b_ptr = c_lib.create_board()
        rec = CaptureRecord()

        # Warmup
        for _ in range(1000):
            c_lib.place_stone_c(b_ptr, 9, 9, 0, ctypes.byref(rec))
            c_lib.undo_stone_c(b_ptr, 9, 9, 0, ctypes.byref(rec))

        # Benchmark
        N = 200_000
        t0 = time.perf_counter()
        for i in range(N):
            r = (i % 15) + 2
            c = ((i * 7) % 15) + 2
            c_lib.place_stone_c(b_ptr, r, c, 0, ctypes.byref(rec))
            c_lib.undo_stone_c(b_ptr, r, c, 0, ctypes.byref(rec))
        t1 = time.perf_counter()
        c_lib.free_board(b_ptr)

        dt = t1 - t0
        c_ns = (dt / (2 * N)) * 1e9
        c_speed = (2 * N) / dt
        print(f"[1] Native C++ Engine (C++17 -O3 -march=native):")
        print(f"    Latency: {c_ns:.1f} ns / op | Throughput: {c_speed/1e6:.2f} Million ops/sec")
    except Exception as e:
        print(f"[1] Native C++ Library test error: {e}")
else:
    print("[1] Native C++ Library: libomok_engine.so not found at tested paths.")

# -------------------------------------------------------------
# 2. Numba LLVM JIT FastBoard
# -------------------------------------------------------------
fb = FastBoard()
# Warmup JIT
for _ in range(100):
    num_caps, buf = fb.place_stone(9, 9, 0)
    fb.undo_stone(9, 9, 0, num_caps, buf)

N = 200_000
t0 = time.perf_counter()
for i in range(N):
    r = (i % 15) + 2
    c = ((i * 7) % 15) + 2
    num_caps, buf = fb.place_stone(r, c, 0)
    fb.undo_stone(r, c, 0, num_caps, buf)
t1 = time.perf_counter()
dt = t1 - t0
fast_ns = (dt / (2 * N)) * 1e9
fast_speed = (2 * N) / dt
print(f"\n[2] Numba LLVM JIT Engine (FastBoard):")
print(f"    Latency: {fast_ns:.1f} ns / op | Throughput: {fast_speed/1e6:.2f} Million ops/sec")

# -------------------------------------------------------------
# 3. Standard Python Board
# -------------------------------------------------------------
py_b = Board()
N_py = 20_000
t0 = time.perf_counter()
for i in range(N_py):
    py_b_copy = py_b.copy()
    py_b_copy.place_stone(9, 9, 0)
t1 = time.perf_counter()
dt = t1 - t0
py_ns = (dt / N_py) * 1e9
py_speed = N_py / dt
print(f"\n[3] Standard Python Board (copy + place):")
print(f"    Latency: {py_ns/1e3:.2f} us / op | Throughput: {py_speed:,.0f} ops/sec")
print(f"    >>> C++ / Numba Speedup over Python: {fast_speed / py_speed:.1f}x faster!")

# -------------------------------------------------------------
# 4. Neural Network Inference on RTX 5090
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[4] Neural Network Inference on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}):")

# Test 1x baseline (5 blocks, 64 channels)
m1 = PolicyValueNet.create_baseline().to(device).eval()
x1 = torch.randn(64, 7, 19, 19, device=device)
for _ in range(10): _ = m1(x1)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50): _ = m1(x1)
torch.cuda.synchronize()
t1 = time.perf_counter()
lat1 = (t1 - t0) / 50 * 1000
fps1 = (64 * 50) / (t1 - t0)
print(f"    [Baseline 5-Block Model] (5.29M params): {lat1:.2f} ms / batch (64) | {fps1:,.0f} states/sec")

# Test 2x scaled model (10 blocks, 128 channels)
m2 = PolicyValueNet.create_2x().to(device).eval()
x2 = torch.randn(64, 7, 19, 19, device=device)
for _ in range(10): _ = m2(x2)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50): _ = m2(x2)
torch.cuda.synchronize()
t1 = time.perf_counter()
lat2 = (t1 - t0) / 50 * 1000
fps2 = (64 * 50) / (t1 - t0)
print(f"    [Scaled 2x 10-Block Model] (7.88M params, 8x backbone): {lat2:.2f} ms / batch (64) | {fps2:,.0f} states/sec")

print("=" * 70)
print("BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 70)
