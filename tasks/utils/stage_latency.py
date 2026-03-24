import time
from collections import defaultdict
from contextlib import contextmanager


def _get_gpu_snapshot():
    """Return current GPU memory stats per device and totals (MB). No-op if CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        snapshot = {}
        total_allocated = 0.0
        total_reserved = 0.0
        total_max_allocated = 0.0
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
            snapshot[f"device_{i}"] = {
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "max_allocated_mb": round(max_allocated, 2),
            }
            total_allocated += allocated
            total_reserved += reserved
            total_max_allocated += max_allocated
        snapshot["total_allocated_mb"] = round(total_allocated, 2)
        snapshot["total_reserved_mb"] = round(total_reserved, 2)
        snapshot["total_max_allocated_mb"] = round(total_max_allocated, 2)
        return snapshot
    except Exception:
        return None


@contextmanager
def cuda_timer(sync: bool = True):
    try:
        import torch
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    start = time.perf_counter()
    yield

    try:
        import torch
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    end = time.perf_counter()
    print(f"[TIMER] {end - start:.6f}s")


class StageLatencyProfiler:
    """
    Stage-wise latency profiler for VLM pipeline.
    Records per-stage times and GPU memory (max_allocated per sample, last snapshot).
    """

    def __init__(self, use_cuda_timer: bool = False, track_gpu: bool = True):
        self.records = defaultdict(list)  # stage -> list of elapsed times
        self.gpu_max_per_sample = defaultdict(list)  # stage -> list of max_allocated_mb (per sample)
        self.gpu_last_snapshot = {}  # stage -> last gpu snapshot dict
        self.use_cuda_timer = use_cuda_timer
        self.track_gpu = track_gpu

    def measure(self, stage_name: str, fn):
        print(f"[STAGE] {stage_name}")

        try:
            import torch
            if self.track_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        if self.use_cuda_timer:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

            start = time.perf_counter()
            result = fn()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            elapsed = time.perf_counter() - start
        else:
            start = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - start

        self.records[stage_name].append(elapsed)

        if self.track_gpu:
            snapshot = _get_gpu_snapshot()
            if snapshot is not None:
                self.gpu_last_snapshot[stage_name] = snapshot
                total_max_mb = snapshot.get("total_max_allocated_mb", 0)
                self.gpu_max_per_sample[stage_name].append(total_max_mb)

        return result

    def to_dict(self):
        out = {}
        for stage, times in self.records.items():
            n = len(times)
            entry = {
                "count": n,
                "avg": sum(times) / n if n else 0,
                "min": min(times) if n else 0,
                "max": max(times) if n else 0,
                "times": times,
            }
            gpu_samples = self.gpu_max_per_sample.get(stage, [])
            if gpu_samples:
                entry["gpu_max_allocated_mb_avg"] = round(sum(gpu_samples) / len(gpu_samples), 2)
                entry["gpu_max_allocated_mb_max"] = round(max(gpu_samples), 2)
                entry["gpu_max_allocated_mb_samples"] = [round(x, 2) for x in gpu_samples]
            if stage in self.gpu_last_snapshot:
                entry["gpu_last_sample"] = self.gpu_last_snapshot[stage]
            out[stage] = entry
        return out
