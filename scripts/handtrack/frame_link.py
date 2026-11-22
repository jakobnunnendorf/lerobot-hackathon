import argparse
import struct
import subprocess
import sys
import time
from multiprocessing import shared_memory

import numpy as np

HEADER = struct.Struct("<QQ")  # sequence number, capture time in perf_counter_ns
SHAPE = (480, 640, 3)


class FrameWriter:
    # Lives in the robot program. Writing never waits for the reader, so a slow or crashed tracker
    # cannot stall the 30 Hz control loop (a blocking socket send could).
    def __init__(self, name, shape=SHAPE):
        size = HEADER.size + int(np.prod(shape))
        self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        self.frame = np.ndarray(shape, np.uint8, self.shm.buf, offset=HEADER.size)
        self.sequence = 0

    def write(self, frame, captured_ns):
        self.sequence += 1
        HEADER.pack_into(self.shm.buf, 0, 2 * self.sequence - 1, captured_ns)
        self.frame[...] = frame
        HEADER.pack_into(self.shm.buf, 0, 2 * self.sequence, captured_ns)

    def close(self):
        del self.frame
        self.shm.close()
        self.shm.unlink()


class FrameReader:
    def __init__(self, name, shape=SHAPE):
        # On Linux with Python 3.12 the reader's resource tracker unlinks the block when the reader exits
        # (fixed by track=False in 3.13), so unregister it there; Windows has no resource tracker for this.
        self.shm = shared_memory.SharedMemory(name=name)
        if sys.platform != "win32":
            from multiprocessing import resource_tracker
            resource_tracker.unregister(self.shm._name, "shared_memory")
        self.frame = np.ndarray(shape, np.uint8, self.shm.buf, offset=HEADER.size)
        self.last = 0

    def latest(self):
        # Seqlock: an odd sequence number means the writer is mid-copy; retry until a clean copy.
        while True:
            before, captured_ns = HEADER.unpack_from(self.shm.buf, 0)
            if before == self.last or before % 2:
                return None
            copy = self.frame.copy()
            after, _ = HEADER.unpack_from(self.shm.buf, 0)
            if after == before:
                self.last = before
                return copy, captured_ns

    def close(self):
        del self.frame
        self.shm.close()


def run_reader(name, seconds):
    reader = FrameReader(name)
    latencies, end = [], time.perf_counter() + seconds
    while time.perf_counter() < end:
        got = reader.latest()
        if got is None:
            time.sleep(0.001)
            continue
        frame, captured_ns = got
        latencies.append((time.perf_counter_ns() - captured_ns) / 1e6)
    reader.close()
    latencies.sort()
    print(f"reader: {len(latencies)} frames, latency median {latencies[len(latencies) // 2]:.2f} ms, "
          f"p95 {latencies[int(0.95 * len(latencies))]:.2f} ms, max {latencies[-1]:.2f} ms")


def run_writer(name, seconds):
    writer = FrameWriter(name)
    reader = subprocess.Popen([sys.executable, "-I", __file__, "--reader", "--name", name, "--seconds", str(seconds)])
    time.sleep(0.5)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, SHAPE, np.uint8) for _ in range(4)]
    costs, end, i = [], time.perf_counter() + seconds - 1, 0
    while time.perf_counter() < end:
        t0 = time.perf_counter_ns()
        writer.write(frames[i % 4], t0)
        costs.append((time.perf_counter_ns() - t0) / 1e6)
        i += 1
        time.sleep(1 / 30)
    reader.wait()
    writer.close()
    costs.sort()
    print(f"writer: {len(costs)} frames at 30 Hz, cost per write median {costs[len(costs) // 2]:.3f} ms, max {costs[-1]:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", action="store_true")
    parser.add_argument("--name", default="top_camera_frames")
    parser.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    (run_reader if args.reader else run_writer)(args.name, args.seconds)
