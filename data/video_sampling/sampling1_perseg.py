"""
Video 전체에 대해 총 num_frames개 프레임을 뽑는 샘플링.

- Segment당 가운데 시간에서 1개만 사용.
- Video 전체 기준으로 총 num_frames개 타임스탬프(초) 반환.
"""

from __future__ import annotations

from typing import List, Tuple

Segment = Tuple[float, float]


def get_video_timestamps_one_per_segment(
    duration: float,
    segments: List[Segment],
    num_frames: int = 4,
    fps: float = 30.0,
) -> List[float]:
    """
    Video 전체에 대해 총 num_frames개의 타임스탬프(초)를 반환한다.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")
    if duration <= 0:
        duration = 1.0

    if not segments:
        return _uniform_fallback(duration, num_frames)

    segs: List[Tuple[float, float]] = [(float(s), float(e)) for s, e in segments]
    first_start = min(s[0] for s in segs)
    has_normal = first_start > 1e-6
    target_anomaly = num_frames - 1 if has_normal else num_frames

    anomaly_seconds = _pick_midpoints_one_per_segment(segs, target_anomaly)
    frame_indices: List[int] = [int(round(s * fps)) for s in anomaly_seconds]

    if has_normal:
        normal_time = min(duration * 0.1, max(0.0, first_start - 0.01))
        normal_idx = int(round(normal_time * fps))
        frame_indices.insert(0, normal_idx)

    frame_indices = _dedupe_and_cap(frame_indices, num_frames)
    max_fi = max(0, int(duration * fps))
    if len(frame_indices) < num_frames:
        frame_indices = _fill_from_longest_gaps(
            frame_indices, segs, duration, fps, num_frames
        )
    frame_indices.sort()
    out_seconds = [fi / fps for fi in frame_indices]
    return out_seconds[:num_frames]


def _uniform_fallback(duration: float, num_frames: int) -> List[float]:
    step = duration / (num_frames + 1)
    return [step * (i + 1) for i in range(num_frames)]


def _pick_midpoints_one_per_segment(
    segments: List[Segment], target: int
) -> List[float]:
    n = len(segments)
    lengths = [(e - s, (s, e)) for s, e in segments]
    lengths.sort(key=lambda x: -x[0])

    if n >= target:
        selected = [lengths[i][1] for i in range(target)]
        return [(s + e) / 2.0 for s, e in selected]
    mids = [(s + e) / 2.0 for s, e in segments]
    return mids


def _dedupe_and_cap(frame_indices: List[int], cap: int) -> List[int]:
    seen = set()
    out: List[int] = []
    for fi in frame_indices:
        if fi not in seen:
            seen.add(fi)
            out.append(fi)
    while len(out) > cap:
        out.pop()
    return out


def _gaps_outside_segments(
    segments: List[Segment], duration: float
) -> List[Tuple[float, float]]:
    if not segments:
        return [(0.0, duration)] if duration > 0 else []
    segs = sorted(segments, key=lambda x: x[0])
    gaps: List[Tuple[float, float]] = []
    if segs[0][0] > 1e-6:
        gaps.append((0.0, segs[0][0]))
    for i in range(len(segs) - 1):
        a_end, b_start = segs[i][1], segs[i + 1][0]
        if b_start - a_end > 1e-6:
            gaps.append((a_end, b_start))
    if duration - segs[-1][1] > 1e-6:
        gaps.append((segs[-1][1], duration))
    return gaps


def _fill_from_longest_gaps(
    frame_indices: List[int],
    segments: List[Segment],
    duration: float,
    fps: float,
    num_frames: int,
) -> List[int]:
    out = list(frame_indices)
    existing = set(out)
    need = num_frames - len(out)
    if need <= 0 or fps <= 0:
        return out[:num_frames]
    max_fi = int(duration * fps)
    gaps = _gaps_outside_segments(segments, duration)
    gaps_with_len = [(e - s, (s, e)) for s, e in gaps if e > s]
    gaps_with_len.sort(key=lambda x: -x[0])
    for _len, (s, e) in gaps_with_len:
        if need <= 0:
            break
        mid_sec = (s + e) / 2.0
        fi = int(round(mid_sec * fps))
        fi = max(0, min(fi, max_fi))
        if fi not in existing:
            existing.add(fi)
            out.append(fi)
            need -= 1
    if need > 0:
        out = _fill_to_num_frames(out, max_fi, num_frames)
    out.sort()
    return out[:num_frames]


def _fill_to_num_frames(
    frame_indices: List[int], max_fi: int, num_frames: int
) -> List[int]:
    out = list(frame_indices)
    existing = set(out)
    need = num_frames - len(out)
    if need <= 0 or max_fi < 0:
        return out[:num_frames]
    candidates = [i for i in range(0, max_fi + 1) if i not in existing]
    if len(candidates) < need:
        step = 1
    else:
        step = max(1, len(candidates) // (need + 1))
    added = 0
    for i in range(0, len(candidates), step):
        if added >= need:
            break
        fi = candidates[i]
        if fi not in existing:
            existing.add(fi)
            out.append(fi)
            added += 1
    out.sort()
    return out[:num_frames]

