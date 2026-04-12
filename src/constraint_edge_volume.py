from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from k3m_codec import K3MCodec, MemoryQuery, SyntheticEvent, TubeRecord


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


@dataclass
class ConstraintEdgeRecord:
    step_idx: int
    source_state: str
    edge_type: str
    target: str
    weight: float
    trigger: str = ""
    status: str = ""

    def token(self) -> str:
        parts = [self.edge_type, self.source_state, self.target]
        if self.trigger:
            parts.append(f"trigger={self.trigger}")
        if self.status:
            parts.append(f"status={self.status}")
        return "::".join(parts)


@dataclass
class ProjectedConstraintPattern:
    step_idx: int
    address_token: str
    weight: float
    support_mask: np.ndarray
    center: Tuple[int, int, int]
    kind: str


class ConstraintEdgeVolumeProjector:
    def __init__(
        self,
        shape: Tuple[int, int, int] = (24, 24, 24),
        tube_half_length: int = 2,
        tube_radius: int = 1,
        junction_radius: int = 1,
    ) -> None:
        self.shape = tuple(int(dim) for dim in shape)
        self.tube_half_length = int(tube_half_length)
        self.tube_radius = int(tube_radius)
        self.junction_radius = int(junction_radius)
        self._render_codec = K3MCodec(
            tube_half_length=self.tube_half_length,
            tube_radius=self.tube_radius,
            max_tubes=1,
            max_junctions=0,
            residual_budget=0,
        )
        self.direction_bank = list(self._render_codec.direction_bank)

    def project(
        self,
        edge_records: Sequence[ConstraintEdgeRecord],
        total_steps: int,
    ) -> Tuple[np.ndarray, List[ProjectedConstraintPattern], Dict[str, float]]:
        volume = np.zeros(self.shape, dtype=np.float32)
        patterns: List[ProjectedConstraintPattern] = []
        for record in edge_records:
            contribution, mask, center = self._render_record(record, total_steps)
            volume += contribution
            patterns.append(
                ProjectedConstraintPattern(
                    step_idx=record.step_idx,
                    address_token=record.token(),
                    weight=float(record.weight),
                    support_mask=mask,
                    center=center,
                    kind=record.edge_type,
                )
            )
        junction_patterns = self._render_junction_patterns(edge_records, total_steps)
        for contribution, pattern in junction_patterns:
            volume += contribution
            patterns.append(pattern)
        tt = np.linspace(0.0, 1.0, self.shape[0], dtype=np.float32)[:, None, None]
        volume += np.float32(0.025) * tt
        stats = {
            "pattern_count": float(len(edge_records)),
            "junction_pattern_count": float(len(junction_patterns)),
            "raw_volume_l1": float(np.sum(np.abs(volume))),
        }
        return volume, patterns, stats

    def _render_record(
        self,
        record: ConstraintEdgeRecord,
        total_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        direction = self._direction(record)
        center = (
            self._time_center(record.step_idx, total_steps),
            self._u_center(record),
            self._v_center(record),
        )
        amplitude = self._amplitude(record)
        tube_record = self._tube_record(
            SyntheticEvent(
                event_id=record.token(),
                direction=direction,
                center=center,
                half_length=self._half_length(record),
                radius=self._radius(record),
                modality="text",
                amplitude=amplitude,
                query=MemoryQuery(
                    direction=tuple(float(item) for item in direction),
                    time_range=(
                        max(0, center[0] - self._half_length(record)),
                        min(self.shape[0] - 1, center[0] + self._half_length(record)),
                    ),
                    modality="text",
                ),
            )
        )
        contribution, mask = self._render_codec._render_tube(self.shape, tube_record)
        return contribution, mask.astype(bool), center

    def _render_junction_patterns(
        self,
        edge_records: Sequence[ConstraintEdgeRecord],
        total_steps: int,
    ) -> List[Tuple[np.ndarray, ProjectedConstraintPattern]]:
        grouped: Dict[Tuple[int, str], List[ConstraintEdgeRecord]] = defaultdict(list)
        for record in edge_records:
            grouped[(record.step_idx, record.source_state)].append(record)
        rendered: List[Tuple[np.ndarray, ProjectedConstraintPattern]] = []
        for (step_idx, source_state), members in grouped.items():
            if len(members) < 2:
                continue
            center = (
                self._time_center(step_idx, total_steps),
                int(round(sum(self._u_center(item) for item in members) / len(members))),
                int(round(sum(self._v_center(item) for item in members) / len(members))),
            )
            contribution = np.zeros(self.shape, dtype=np.float32)
            mask = np.zeros(self.shape, dtype=bool)
            t0, u0, v0 = center
            for dt in range(-self.junction_radius, self.junction_radius + 1):
                for du in range(-self.junction_radius, self.junction_radius + 1):
                    for dv in range(-self.junction_radius, self.junction_radius + 1):
                        tt = min(max(t0 + dt, 0), self.shape[0] - 1)
                        uu = min(max(u0 + du, 0), self.shape[1] - 1)
                        vv = min(max(v0 + dv, 0), self.shape[2] - 1)
                        contribution[tt, uu, vv] += np.float32(0.55 + 0.1 * len(members))
                        mask[tt, uu, vv] = True
            token = f"junction::{source_state}::degree[{len(members)}]"
            rendered.append(
                (
                    contribution,
                    ProjectedConstraintPattern(
                        step_idx=step_idx,
                        address_token=token,
                        weight=float(1.0 + 0.15 * len(members)),
                        support_mask=mask,
                        center=center,
                        kind="junction",
                    ),
                )
            )
        return rendered

    def _time_center(self, step_idx: int, total_steps: int) -> int:
        if total_steps <= 1:
            return self.shape[0] // 2
        return int(round(step_idx * (self.shape[0] - 1) / max(total_steps - 1, 1)))

    def _u_center(self, record: ConstraintEdgeRecord) -> int:
        key = f"u::{record.source_state}::{record.trigger or record.edge_type}"
        return stable_hash(key, self.shape[1])

    def _v_center(self, record: ConstraintEdgeRecord) -> int:
        key = f"v::{record.target}::{record.edge_type}::{record.status}"
        return stable_hash(key, self.shape[2])

    def _direction(self, record: ConstraintEdgeRecord) -> Tuple[int, int, int]:
        idx = stable_hash(f"dir::{record.token()}", len(self.direction_bank))
        return self.direction_bank[idx]

    def _amplitude(self, record: ConstraintEdgeRecord) -> float:
        base = {
            "allow": 1.5,
            "block": 1.7,
            "require": 1.4,
            "trigger": 1.3,
            "step": 1.5,
            "flow": 1.6,
            "conflict": 1.8,
            "junction": 1.2,
        }.get(record.edge_type, 1.2)
        if record.status == "realized":
            base += 0.35
        if record.status == "blocked":
            base += 0.45
        if record.status == "outside":
            base += 0.20
        return float(base * max(record.weight, 0.3))

    def _half_length(self, record: ConstraintEdgeRecord) -> int:
        if record.edge_type in {"flow", "step", "conflict"}:
            return min(self.tube_half_length + 1, self.shape[0] // 4)
        return self.tube_half_length

    def _radius(self, record: ConstraintEdgeRecord) -> int:
        if record.edge_type in {"block", "conflict"}:
            return self.tube_radius + 1
        return self.tube_radius

    def _tube_record(self, event: SyntheticEvent) -> TubeRecord:
        scale = max(abs(event.amplitude) / 32767.0, 1e-8)
        q_signal = np.full(
            2 * event.half_length + 1,
            int(round(event.amplitude / scale)),
            dtype=np.int16,
        )
        return TubeRecord(
            tube_id=0,
            direction_idx=0,
            direction=event.direction,
            center=event.center,
            half_length=event.half_length,
            radius=event.radius,
            time_range=(
                max(0, event.center[0] - event.half_length),
                min(self.shape[0] - 1, event.center[0] + event.half_length),
            ),
            modality=event.modality,
            importance=min(max(abs(event.amplitude), 0.2), 4.0),
            scale=scale,
            q_signal=q_signal,
            support_size=0,
        )
