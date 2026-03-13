import time
import torch
from collections import defaultdict
from contextlib import contextmanager


@contextmanager
def cuda_timer(sync: bool = True):
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    yield

    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    print(f"[TIMER] {end - start:.6f}s")


class StageLatencyProfiler:
    """
    Stage-wise latency profiler for VLM pipeline.
    """

    def __init__(self, use_cuda_timer: bool = False):
        self.records = defaultdict(list)
        self.use_cuda_timer = use_cuda_timer

    def measure(self, stage_name: str, fn):
        print(f"[STAGE] {stage_name}")

        if self.use_cuda_timer:
            with cuda_timer():
                result = fn()
            start = time.time()
            elapsed = time.time() - start
        else:
            start = time.time()
            result = fn()
            elapsed = time.time() - start

        self.records[stage_name].append(elapsed)
        return result

    def to_dict(self):
        return {
            stage: {
                "count": len(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "times": times,
            }
            for stage, times in self.records.items()
        }

