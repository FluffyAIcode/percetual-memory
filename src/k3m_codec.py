from __future__ import annotations

from dataclasses import dataclass, field
import math
import pickle
import time
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Array3D = np.ndarray


def _normalize(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr) + 1e-12
    return arr / norm


def _shift_zero(arr: Array3D, shift: Tuple[int, int, int]) -> Array3D:
    out = np.zeros_like(arr)
    src_slices = []
    dst_slices = []
    for axis, delta in enumerate(shift):
        size = arr.shape[axis]
        if delta >= 0:
            src = slice(0, size - delta)
            dst = slice(delta, size)
        else:
            src = slice(-delta, size)
            dst = slice(0, size + delta)
        src_slices.append(src)
        dst_slices.append(dst)
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def _moving_average_axis(arr: Array3D, radius: int, axis: int) -> Array3D:
    if radius <= 0:
        return arr.copy()
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(arr, pad, mode="edge")
    width = 2 * radius + 1
    kernel = np.ones(width, dtype=np.float32) / np.float32(width)
    return np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="valid"),
        axis,
        padded,
    ).astype(np.float32)


def box_blur3d(arr: Array3D, radii: Tuple[int, int, int]) -> Array3D:
    out = arr.astype(np.float32, copy=True)
    for axis, radius in enumerate(radii):
        out = _moving_average_axis(out, radius=radius, axis=axis)
    return out


def quantize_array(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = max(max_abs / 32767.0, 1e-8)
    q = np.round(arr / scale).astype(np.int16)
    return q, scale


def dequantize_array(q: np.ndarray, scale: float) -> np.ndarray:
    return (q.astype(np.float32) * np.float32(scale)).astype(np.float32)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = _normalize(a)
    vb = _normalize(b)
    return float(np.clip(np.dot(va, vb), -1.0, 1.0))


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-12:
        return 120.0
    span = float(np.max(x) - np.min(x))
    span = max(span, 1e-6)
    return 20.0 * math.log10(span / math.sqrt(mse))


def _serialize_size_bytes(obj: object) -> int:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return len(zlib.compress(payload, level=9))


@dataclass
class TubeRecord:
    tube_id: int
    direction_idx: int
    direction: Tuple[int, int, int]
    center: Tuple[int, int, int]
    half_length: int
    radius: int
    time_range: Tuple[int, int]
    modality: str
    importance: float
    scale: float
    q_signal: np.ndarray
    support_size: int


@dataclass
class JunctionRecord:
    junction_id: int
    center: Tuple[int, int, int]
    patch_radius: int
    incident_tube_ids: List[int]
    importance: float
    scale: float
    q_patch: np.ndarray


@dataclass
class SparseResidual:
    indices: np.ndarray
    q_values: np.ndarray
    scale: float


@dataclass
class K3MArchive:
    shape: Tuple[int, int, int]
    base_q: np.ndarray
    base_scale: float
    tubes: List[TubeRecord]
    junctions: List[JunctionRecord]
    residual: SparseResidual
    direction_bank: List[Tuple[int, int, int]]
    metadata: Dict[str, float] = field(default_factory=dict)

    def compressed_size_bytes(self) -> int:
        payload = {
            "shape": self.shape,
            "base_q": self.base_q,
            "base_scale": self.base_scale,
            "tubes": self.tubes,
            "junctions": self.junctions,
            "residual": self.residual,
            "direction_bank": self.direction_bank,
            "metadata": self.metadata,
        }
        return _serialize_size_bytes(payload)


@dataclass
class MemoryQuery:
    direction: Tuple[float, float, float]
    time_range: Optional[Tuple[int, int]] = None
    modality: Optional[str] = None


@dataclass
class SyntheticEvent:
    event_id: str
    direction: Tuple[int, int, int]
    center: Tuple[int, int, int]
    half_length: int
    radius: int
    modality: str
    amplitude: float
    query: MemoryQuery


@dataclass
class DatasetExample:
    name: str
    volume: np.ndarray
    events: List[SyntheticEvent]


class K3MCodec:
    """Prototype support-first codec with a discrete Kakeya-like constructor.

    The implementation first builds a thin active support by selecting
    direction-aligned tubes and overlap junctions, then encodes signal
    content restricted to that support together with a background field
    and a sparse residual.
    """

    def __init__(
        self,
        base_radii: Tuple[int, int, int] = (2, 2, 2),
        tube_half_length: int = 4,
        tube_radius: int = 1,
        max_tubes: int = 14,
        max_junctions: int = 10,
        residual_budget: int = 512,
        response_steps: int = 4,
    ) -> None:
        self.base_radii = base_radii
        self.tube_half_length = tube_half_length
        self.tube_radius = tube_radius
        self.max_tubes = max_tubes
        self.max_junctions = max_junctions
        self.residual_budget = residual_budget
        self.response_steps = response_steps
        self.min_active_dirs = 4
        self.max_active_dirs = 8
        self.coverage_bonus = 0.9
        self.volume_penalty_power = 0.35
        self.direction_bank = [
            (1, 0, 0),
            (1, 1, 0),
            (1, -1, 0),
            (1, 0, 1),
            (1, 0, -1),
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (0, 1, 0),
            (0, 0, 1),
            (0, 1, 1),
            (0, 1, -1),
        ]

    def encode(self, volume: Array3D) -> K3MArchive:
        """Encode by constructing support first, then coding on support."""
        volume = volume.astype(np.float32, copy=False)
        base = box_blur3d(volume, self.base_radii)
        residual = volume - base

        active_direction_indices = self._detect_active_directions(residual)
        tubes = self._select_tubes(residual, active_direction_indices)
        residual_after_tubes = residual.copy()
        support_count = np.zeros(volume.shape, dtype=np.uint16)
        support_mask = np.zeros(volume.shape, dtype=bool)

        for tube in tubes:
            contribution, mask = self._render_tube(volume.shape, tube)
            residual_after_tubes -= contribution
            support_count += mask.astype(np.uint16)
            support_mask |= mask

        junctions = self._extract_junctions(residual_after_tubes, support_count, tubes)
        residual_after_junctions = residual_after_tubes.copy()
        for junction in junctions:
            j_recon, j_mask = self._render_junction(volume.shape, junction, return_mask=True)
            residual_after_junctions -= j_recon
            support_mask |= j_mask

        sparse_residual = self._encode_sparse_residual(residual_after_junctions)
        base_q, base_scale = quantize_array(base)
        support_ratio = float(np.mean(support_mask))
        covered_direction_indices = sorted({tube.direction_idx for tube in tubes})
        coverage_ratio = float(
            len(set(covered_direction_indices) & set(active_direction_indices))
            / max(len(active_direction_indices), 1)
        )

        metadata = {
            "support_ratio": support_ratio,
            "n_tubes": float(len(tubes)),
            "n_junctions": float(len(junctions)),
            "support_constructor": "discrete_greedy_kakeya_like",
            "active_direction_count": float(len(active_direction_indices)),
            "covered_direction_count": float(len(covered_direction_indices)),
            "direction_coverage_ratio": coverage_ratio,
        }
        return K3MArchive(
            shape=tuple(volume.shape),
            base_q=base_q,
            base_scale=base_scale,
            tubes=tubes,
            junctions=junctions,
            residual=sparse_residual,
            direction_bank=self.direction_bank,
            metadata=metadata,
        )

    def decode(self, archive: K3MArchive) -> Array3D:
        recon = dequantize_array(archive.base_q, archive.base_scale)
        for tube in archive.tubes:
            contrib, _ = self._render_tube(archive.shape, tube)
            recon += contrib
        for junction in archive.junctions:
            recon += self._render_junction(archive.shape, junction)
        if archive.residual.indices.size:
            values = dequantize_array(archive.residual.q_values, archive.residual.scale)
            idx = archive.residual.indices
            recon[idx[:, 0], idx[:, 1], idx[:, 2]] += values
        return recon.astype(np.float32)

    def query_decode(
        self,
        archive: K3MArchive,
        query: MemoryQuery,
        top_k: int = 3,
        margin: int = 2,
    ) -> Tuple[np.ndarray, List[int], Tuple[slice, slice, slice]]:
        ranked = self.retrieve_tubes(archive, query, top_k=top_k)
        if not ranked:
            empty = np.zeros((1, 1, 1), dtype=np.float32)
            return empty, [], (slice(0, 1), slice(0, 1), slice(0, 1))

        coords = []
        for tube in ranked:
            for point in self._tube_line_points(archive.shape, tube):
                coords.append(point)
        arr = np.array(coords, dtype=np.int32)
        lo = np.maximum(arr.min(axis=0) - margin, 0)
        hi = np.minimum(arr.max(axis=0) + margin + 1, np.array(archive.shape))
        region = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        full = self.decode(archive)
        return full[region], [tube.tube_id for tube in ranked], region

    def retrieve_tubes(
        self, archive: K3MArchive, query: MemoryQuery, top_k: int = 3
    ) -> List[TubeRecord]:
        scores = []
        for tube in archive.tubes:
            dir_score = abs(cosine_similarity(query.direction, tube.direction))
            time_score = 1.0
            if query.time_range is not None:
                q0, q1 = query.time_range
                t0, t1 = tube.time_range
                overlap = max(0, min(q1, t1) - max(q0, t0) + 1)
                span = max(q1 - q0 + 1, 1)
                time_score = overlap / span
            modality_score = 1.0
            if query.modality is not None and query.modality != tube.modality:
                modality_score = 0.4
            score = dir_score * (0.6 + 0.4 * time_score) * modality_score
            score *= 0.4 + 0.6 * min(tube.importance, 1.0)
            scores.append((score, tube))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [tube for score, tube in scores[:top_k] if score > 0.05]

    def _direction_response(self, residual: Array3D, direction: Tuple[int, int, int]) -> Array3D:
        energy = residual ** 2
        response = np.zeros_like(energy)
        for step in range(-self.response_steps, self.response_steps + 1):
            shift = tuple(step * d for d in direction)
            response += _shift_zero(energy, shift)
        return response

    def _detect_active_directions(self, residual: Array3D) -> List[int]:
        """Detect a small active direction family before support construction."""
        direction_scores = []
        for direction_idx, direction in enumerate(self.direction_bank):
            response = self._direction_response(residual, direction)
            direction_scores.append((float(np.max(response)), direction_idx))
        direction_scores.sort(reverse=True)

        max_score = max(direction_scores[0][0], 1e-8)
        active = [
            direction_idx
            for score, direction_idx in direction_scores
            if score >= 0.25 * max_score
        ]
        if len(active) < self.min_active_dirs:
            active = [direction_idx for _, direction_idx in direction_scores[: self.min_active_dirs]]
        active = active[: self.max_active_dirs]
        return sorted(active)

    def _select_tubes(
        self, residual: Array3D, active_direction_indices: Optional[List[int]] = None
    ) -> List[TubeRecord]:
        """Greedy discrete Kakeya-like support constructor over a direction bank."""
        working = residual.copy()
        tubes: List[TubeRecord] = []
        initial_energy = float(np.sum(working ** 2)) + 1e-8
        direction_counts = [0 for _ in self.direction_bank]
        if active_direction_indices is None:
            active_direction_indices = list(range(len(self.direction_bank)))
        uncovered = set(active_direction_indices)

        for tube_id in range(self.max_tubes):
            best = None
            best_direction_idx = -1
            for direction_idx in active_direction_indices:
                direction = self.direction_bank[direction_idx]
                response = self._direction_response(working, direction)
                flat_idx = int(np.argmax(response))
                raw_score = float(response.reshape(-1)[flat_idx])
                center = np.unravel_index(flat_idx, working.shape)
                approx_volume = max(
                    1.0,
                    len(self._tube_line_points(working.shape, None, center, direction))
                    * (2 * self.tube_radius + 1) ** 2,
                )
                coverage_multiplier = (
                    1.0 + self.coverage_bonus if direction_idx in uncovered else 1.0
                )
                score = (
                    raw_score
                    * coverage_multiplier
                    / ((1.0 + 0.35 * direction_counts[direction_idx]) * (approx_volume ** self.volume_penalty_power))
                )
                if best is None or score > best[0]:
                    best = (score, center)
                    best_direction_idx = direction_idx
            if best is None:
                break
            score, center = best
            if score < 0.01 * initial_energy / np.prod(working.shape):
                break

            tube = self._fit_tube(
                residual=working,
                tube_id=tube_id,
                direction_idx=best_direction_idx,
                center=center,
            )
            if tube.support_size == 0 or np.max(np.abs(tube.q_signal)) == 0:
                break
            contribution, _ = self._render_tube(working.shape, tube)
            working -= contribution
            tubes.append(tube)
            direction_counts[best_direction_idx] += 1
            uncovered.discard(best_direction_idx)
            if not uncovered and len(tubes) >= len(active_direction_indices):
                break

        return tubes

    def _fit_tube(
        self,
        residual: Array3D,
        tube_id: int,
        direction_idx: int,
        center: Tuple[int, int, int],
    ) -> TubeRecord:
        direction = self.direction_bank[direction_idx]
        signal = []
        points = self._tube_line_points(residual.shape, None, center, direction)
        time_coords = []
        support_size = 0
        for point in points:
            patch = self._extract_cube(
                residual, point, radius=self.tube_radius, pad_value=0.0
            )
            signal.append(float(np.mean(patch)))
            support_size += patch.size
            time_coords.append(point[0])
        signal_arr = np.array(signal, dtype=np.float32)
        q_signal, scale = quantize_array(signal_arr)
        modality = self._infer_modality(center, residual.shape)
        importance = float(np.linalg.norm(signal_arr) / (np.linalg.norm(residual) + 1e-8))
        return TubeRecord(
            tube_id=tube_id,
            direction_idx=direction_idx,
            direction=direction,
            center=tuple(int(v) for v in center),
            half_length=self.tube_half_length,
            radius=self.tube_radius,
            time_range=(int(min(time_coords)), int(max(time_coords))),
            modality=modality,
            importance=importance,
            scale=scale,
            q_signal=q_signal,
            support_size=support_size,
        )

    def _extract_junctions(
        self,
        residual: Array3D,
        support_count: np.ndarray,
        tubes: List[TubeRecord],
    ) -> List[JunctionRecord]:
        candidates = np.argwhere(support_count >= 2)
        if candidates.size == 0:
            return []

        energy = residual ** 2
        ranked = sorted(
            ((float(energy[tuple(idx)]), tuple(int(v) for v in idx)) for idx in candidates),
            reverse=True,
        )
        chosen: List[JunctionRecord] = []
        used: List[Tuple[int, int, int]] = []
        for importance, center in ranked:
            if len(chosen) >= self.max_junctions:
                break
            if any(sum(abs(a - b) for a, b in zip(center, prev)) <= 3 for prev in used):
                continue
            patch = self._extract_cube(residual, center, radius=1, pad_value=0.0)
            q_patch, scale = quantize_array(patch)
            incident = [
                tube.tube_id
                for tube in tubes
                if self._point_near_tube(center, residual.shape, tube)
            ]
            chosen.append(
                JunctionRecord(
                    junction_id=len(chosen),
                    center=center,
                    patch_radius=1,
                    incident_tube_ids=incident,
                    importance=importance,
                    scale=scale,
                    q_patch=q_patch,
                )
            )
            used.append(center)
        return chosen

    def _encode_sparse_residual(self, residual: Array3D) -> SparseResidual:
        flat = residual.reshape(-1)
        if flat.size == 0:
            return SparseResidual(
                indices=np.empty((0, 3), dtype=np.int16),
                q_values=np.empty(0, dtype=np.int16),
                scale=1.0,
            )
        budget = min(self.residual_budget, flat.size)
        idx = np.argpartition(np.abs(flat), -budget)[-budget:]
        idx = idx[np.argsort(-np.abs(flat[idx]))]
        coords = np.column_stack(np.unravel_index(idx, residual.shape)).astype(np.int16)
        values = flat[idx].astype(np.float32)
        q_values, scale = quantize_array(values)
        return SparseResidual(indices=coords, q_values=q_values, scale=scale)

    def _render_tube(
        self, shape: Tuple[int, int, int], tube: TubeRecord
    ) -> Tuple[np.ndarray, np.ndarray]:
        recon = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        signal = dequantize_array(tube.q_signal, tube.scale)
        points = self._tube_line_points(shape, tube)
        for value, point in zip(signal, points):
            patch_slices = self._cube_slices(shape, point, tube.radius)
            recon[patch_slices] += value
            mask[patch_slices] = True
        return recon, mask

    def _render_junction(
        self,
        shape: Tuple[int, int, int],
        junction: JunctionRecord,
        return_mask: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        recon = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        patch = dequantize_array(junction.q_patch, junction.scale)
        slices = self._cube_slices(shape, junction.center, junction.patch_radius)
        target_shape = tuple(s.stop - s.start for s in slices)
        patch_slices = tuple(slice(0, n) for n in target_shape)
        recon[slices] += patch[patch_slices]
        mask[slices] = True
        if return_mask:
            return recon, mask
        return recon

    def _tube_line_points(
        self,
        shape: Tuple[int, int, int],
        tube: Optional[TubeRecord] = None,
        center: Optional[Tuple[int, int, int]] = None,
        direction: Optional[Tuple[int, int, int]] = None,
    ) -> List[Tuple[int, int, int]]:
        if tube is not None:
            center = tube.center
            direction = tube.direction
            half_length = tube.half_length
        else:
            assert center is not None and direction is not None
            half_length = self.tube_half_length

        points: List[Tuple[int, int, int]] = []
        for step in range(-half_length, half_length + 1):
            point = tuple(center[i] + step * direction[i] for i in range(3))
            if all(0 <= point[i] < shape[i] for i in range(3)):
                points.append(tuple(int(v) for v in point))
        return points

    def _extract_cube(
        self,
        arr: Array3D,
        center: Tuple[int, int, int],
        radius: int,
        pad_value: float,
    ) -> Array3D:
        slices = self._cube_slices(arr.shape, center, radius)
        patch = np.full((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), pad_value, dtype=np.float32)
        target = arr[slices]
        patch_slices = tuple(slice(0, dim) for dim in target.shape)
        patch[patch_slices] = target
        return patch

    def _cube_slices(
        self, shape: Tuple[int, int, int], center: Tuple[int, int, int], radius: int
    ) -> Tuple[slice, slice, slice]:
        slices = []
        for axis in range(3):
            lo = max(center[axis] - radius, 0)
            hi = min(center[axis] + radius + 1, shape[axis])
            slices.append(slice(lo, hi))
        return tuple(slices)  # type: ignore[return-value]

    def _infer_modality(self, center: Tuple[int, int, int], shape: Tuple[int, int, int]) -> str:
        v = center[2]
        cut1 = shape[2] // 3
        cut2 = 2 * shape[2] // 3
        if v < cut1:
            return "text"
        if v < cut2:
            return "image"
        return "video"

    def _point_near_tube(
        self, point: Tuple[int, int, int], shape: Tuple[int, int, int], tube: TubeRecord
    ) -> bool:
        return any(
            sum(abs(a - b) for a, b in zip(point, lp)) <= 2 * tube.radius + 1
            for lp in self._tube_line_points(shape, tube)
        )


def add_synthetic_tube(
    volume: np.ndarray,
    event: SyntheticEvent,
    noise_scale: float = 0.0,
) -> None:
    shape = volume.shape
    codec = K3MCodec(
        tube_half_length=event.half_length,
        tube_radius=event.radius,
        max_tubes=1,
        residual_budget=0,
    )
    record = TubeRecord(
        tube_id=0,
        direction_idx=0,
        direction=event.direction,
        center=event.center,
        half_length=event.half_length,
        radius=event.radius,
        time_range=(
            max(0, event.center[0] - event.half_length),
            min(shape[0] - 1, event.center[0] + event.half_length),
        ),
        modality=event.modality,
        importance=1.0,
        scale=max(abs(event.amplitude) / 32767.0, 1e-8),
        q_signal=np.full(2 * event.half_length + 1, int(round(event.amplitude / max(abs(event.amplitude) / 32767.0, 1e-8))), dtype=np.int16),
        support_size=0,
    )
    contrib, _ = codec._render_tube(shape, record)
    volume += contrib
    if noise_scale > 0:
        volume += np.random.normal(0.0, noise_scale, size=shape).astype(np.float32)


def make_synthetic_dataset(
    name: str,
    shape: Tuple[int, int, int],
    seed: int = 0,
) -> DatasetExample:
    rng = np.random.default_rng(seed)
    t, u, v = shape
    tt, uu, vv = np.meshgrid(
        np.linspace(0, 1, t, dtype=np.float32),
        np.linspace(0, 1, u, dtype=np.float32),
        np.linspace(0, 1, v, dtype=np.float32),
        indexing="ij",
    )
    base = (
        0.25 * np.sin(2 * np.pi * tt)
        + 0.15 * np.cos(2 * np.pi * uu)
        + 0.10 * np.sin(3 * np.pi * vv)
        + 0.05 * (tt + uu + vv)
    ).astype(np.float32)
    volume = base.copy()
    events: List[SyntheticEvent] = []

    if name == "text_like":
        specs = [
            ((1, 1, 0), (t // 3, u // 4, v // 6), "text", 1.8),
            ((1, -1, 0), (t // 2, 2 * u // 3, v // 5), "text", 1.5),
            ((1, 0, 1), (2 * t // 3, u // 2, v // 4), "text", 1.6),
        ]
    elif name == "image_like":
        specs = [
            ((0, 1, 1), (t // 2, u // 3, v // 3), "image", 1.8),
            ((0, 1, -1), (t // 2, 2 * u // 3, 2 * v // 3), "image", 1.7),
            ((1, 1, 1), (t // 2, u // 2, v // 2), "image", 1.2),
        ]
    else:
        specs = [
            ((1, 1, 0), (t // 4, u // 4, 2 * v // 3), "video", 2.0),
            ((1, 0, 1), (t // 2, 2 * u // 3, 3 * v // 4), "video", 1.7),
            ((1, -1, 1), (3 * t // 4, u // 2, v // 2), "multimodal", 1.5),
        ]

    for idx, (direction, center, modality, amplitude) in enumerate(specs):
        event = SyntheticEvent(
            event_id=f"{name}_{idx}",
            direction=direction,
            center=center,
            half_length=4,
            radius=1,
            modality=modality,
            amplitude=amplitude,
            query=MemoryQuery(
                direction=tuple(float(x) for x in direction),
                time_range=(max(0, center[0] - 4), min(t - 1, center[0] + 4)),
                modality=modality if modality in {"text", "image", "video"} else None,
            ),
        )
        add_synthetic_tube(volume, event)
        events.append(event)

    for event_a, event_b in zip(events[:-1], events[1:]):
        center = tuple((a + b) // 2 for a, b in zip(event_a.center, event_b.center))
        slices = tuple(
            slice(max(c - 1, 0), min(c + 2, shape[i])) for i, c in enumerate(center)
        )
        volume[slices] += 1.3

    volume += rng.normal(0.0, 0.08, size=shape).astype(np.float32)
    return DatasetExample(name=name, volume=volume, events=events)


def evaluate_query_recall(
    codec: K3MCodec, archive: K3MArchive, dataset: DatasetExample, top_k: int = 3
) -> float:
    hits = 0
    for event in dataset.events:
        ranked = codec.retrieve_tubes(archive, event.query, top_k=top_k)
        matched = any(
            abs(cosine_similarity(tube.direction, event.direction)) > 0.8
            and max(
                0,
                min(tube.time_range[1], event.query.time_range[1])  # type: ignore[index]
                - max(tube.time_range[0], event.query.time_range[0])  # type: ignore[index]
                + 1,
            )
            >= 2
            for tube in ranked
        )
        hits += int(matched)
    return hits / max(len(dataset.events), 1)


def benchmark_codec(
    codec: K3MCodec,
    dataset: DatasetExample,
    top_k: int = 3,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    archive = codec.encode(dataset.volume)
    t1 = time.perf_counter()
    recon = codec.decode(archive)
    t2 = time.perf_counter()
    _local, _ids, _region = codec.query_decode(archive, dataset.events[0].query, top_k=top_k)
    t3 = time.perf_counter()

    raw_bytes = dataset.volume.nbytes
    zlib_bytes = len(zlib.compress(dataset.volume.astype(np.float32).tobytes(), level=9))
    archive_bytes = archive.compressed_size_bytes()
    rel_l2 = float(np.linalg.norm(dataset.volume - recon) / (np.linalg.norm(dataset.volume) + 1e-8))

    return {
        "raw_bytes": float(raw_bytes),
        "zlib_raw_bytes": float(zlib_bytes),
        "archive_bytes": float(archive_bytes),
        "compression_ratio_vs_raw": float(raw_bytes / max(archive_bytes, 1)),
        "compression_ratio_vs_zlib": float(zlib_bytes / max(archive_bytes, 1)),
        "relative_l2": rel_l2,
        "psnr_db": float(psnr(dataset.volume, recon)),
        "support_ratio": float(archive.metadata.get("support_ratio", 0.0)),
        "n_tubes": float(len(archive.tubes)),
        "n_junctions": float(len(archive.junctions)),
        "query_recall_at_k": float(evaluate_query_recall(codec, archive, dataset, top_k=top_k)),
        "encode_seconds": float(t1 - t0),
        "decode_seconds": float(t2 - t1),
        "query_decode_seconds": float(t3 - t2),
    }
