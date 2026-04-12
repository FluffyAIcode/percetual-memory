from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from k3m_codec import (
    JunctionRecord,
    K3MCodec,
    SparseResidual,
    TubeRecord,
    _serialize_size_bytes,
    dequantize_array,
    quantize_array,
)
from mempalace.dialect import Dialect


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
STOP_WORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "me",
    "more",
    "most",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "up",
    "use",
    "using",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}
COMMON_VERBS = {
    "adopt",
    "build",
    "buy",
    "change",
    "choose",
    "compare",
    "debug",
    "decide",
    "deploy",
    "discuss",
    "explain",
    "fix",
    "implement",
    "improve",
    "learn",
    "migrate",
    "plan",
    "prefer",
    "refactor",
    "remember",
    "replace",
    "research",
    "review",
    "ship",
    "switch",
    "test",
    "travel",
    "update",
    "use",
    "want",
    "work",
}
FLAG_TO_RELATION = {
    "DECISION": "decision",
    "ORIGIN": "origin",
    "CORE": "principle",
    "PIVOT": "transition",
    "SENSITIVE": "private",
    "GENESIS": "origin",
}


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def stable_hash(text: str, modulo: int) -> int:
    value = 1469598103934665603
    for byte in text.encode("utf-8"):
        value ^= byte
        value *= 1099511628211
        value &= 0xFFFFFFFFFFFFFFFF
    return value % modulo


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def sentence_chunks(text: str, chunk_size: int = 2) -> List[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if len(part.strip()) >= 8]
    if not sentences:
        tokens = tokenize(text)
        if not tokens:
            return []
        token_chunks = [
            " ".join(tokens[i : i + 24]) for i in range(0, len(tokens), 24)
        ]
        return [chunk for chunk in token_chunks if chunk]
    chunks = []
    for idx in range(0, len(sentences), chunk_size):
        chunks.append(". ".join(sentences[idx : idx + chunk_size]))
    return chunks


def average_pool3d(arr: np.ndarray, factors: Tuple[int, int, int]) -> np.ndarray:
    pooled = arr.astype(np.float32, copy=False)
    for axis, factor in enumerate(factors):
        if factor <= 1:
            continue
        shape = list(pooled.shape)
        usable = (shape[axis] // factor) * factor
        slicer = [slice(None)] * pooled.ndim
        slicer[axis] = slice(0, usable)
        pooled = pooled[tuple(slicer)]
        new_shape = list(pooled.shape)
        new_shape[axis] = usable // factor
        new_shape.insert(axis + 1, factor)
        pooled = pooled.reshape(new_shape).mean(axis=axis + 1)
    return pooled.astype(np.float32)


def upsample_nearest(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    out = arr
    for axis, target in enumerate(target_shape):
        repeats = int(math.ceil(target / max(out.shape[axis], 1)))
        out = np.repeat(out, repeats=repeats, axis=axis)
    slices = tuple(slice(0, dim) for dim in target_shape)
    return out[slices].astype(np.float32)


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for dt in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if dt == du == dv == 0:
                    continue
                shifted = np.roll(mask, shift=(dt, du, dv), axis=(0, 1, 2))
                if dt > 0:
                    shifted[:dt, :, :] = False
                elif dt < 0:
                    shifted[dt:, :, :] = False
                if du > 0:
                    shifted[:, :du, :] = False
                elif du < 0:
                    shifted[:, du:, :] = False
                if dv > 0:
                    shifted[:, :, :dv] = False
                elif dv < 0:
                    shifted[:, :, dv:] = False
                out |= shifted
    return out


@dataclass
class SemanticEvent:
    t_idx: int
    u_idx: int
    v_idx: int
    weight: float
    semantic_key: str
    relation_key: str
    chunk_index: int
    chunk_text: str
    flags: Tuple[str, ...] = ()


@dataclass
class QueryRepresentation:
    text: str
    events: List[SemanticEvent]
    volume: np.ndarray
    volume_vector: np.ndarray
    support_vector: np.ndarray
    time_vector: np.ndarray
    lexical_vector: np.ndarray
    shadow_vector: np.ndarray
    semantic_vector: np.ndarray
    relation_vector: np.ndarray
    semantic_anchor_ratio: float
    relation_anchor_ratio: float
    centers: List[Tuple[int, int, int]]


@dataclass
class K3MV2Archive:
    shape: Tuple[int, int, int]
    coarse_shape: Tuple[int, int, int]
    coarse_q: np.ndarray
    coarse_scale: float
    tubes: List[TubeRecord]
    junctions: List[JunctionRecord]
    residual: SparseResidual
    direction_bank: List[Tuple[int, int, int]]
    metadata: Dict[str, float] = field(default_factory=dict)

    def compressed_size_bytes(self) -> int:
        payload = {
            "shape": self.shape,
            "coarse_shape": self.coarse_shape,
            "coarse_q": self.coarse_q,
            "coarse_scale": self.coarse_scale,
            "tubes": self.tubes,
            "junctions": self.junctions,
            "residual": self.residual,
            "direction_bank": self.direction_bank,
            "metadata": self.metadata,
        }
        return _serialize_size_bytes(payload)


class SemanticTextProjectorV2:
    def __init__(
        self,
        shape: Tuple[int, int, int] = (12, 12, 12),
        chunk_sentences: int = 2,
    ) -> None:
        self.shape = shape
        self.time_bins, self.u_bins, self.v_bins = shape
        self.chunk_sentences = chunk_sentences
        # Reserve enough overflow bins so query-side OOV semantics do not collapse
        # into only a couple of coordinates.
        self.u_anchor_count = max(2, self.u_bins // 2)
        self.v_anchor_count = max(2, self.v_bins // 2)
        self.lexical_dim = 512
        self.semantic_dim = 256
        self.relation_dim = 128
        self.u_index: Dict[str, int] = {}
        self.v_index: Dict[str, int] = {}
        self.topic_idf: Dict[str, float] = {}
        self.relation_idf: Dict[str, float] = {}
        self.lexical_idf = np.ones(self.lexical_dim, dtype=np.float32)
        self.dialect = Dialect()

    def fit(self, texts: Sequence[str]) -> None:
        semantic_counts: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        semantic_doc_freq: Counter[str] = Counter()
        relation_doc_freq: Counter[str] = Counter()

        for text in texts:
            seen_semantic = set()
            seen_relation = set()
            for chunk in sentence_chunks(text, chunk_size=self.chunk_sentences):
                entities, topics, flags, _emotions = self._extract_semantics(chunk)
                predicates = self._extract_predicates(chunk, flags)
                semantic_units = entities + topics
                for item in semantic_units[:4]:
                    semantic_counts[item] += 1
                    seen_semantic.add(item)
                for item in predicates[:4]:
                    relation_counts[item] += 1
                    seen_relation.add(item)
            for item in seen_semantic:
                semantic_doc_freq[item] += 1
            for item in seen_relation:
                relation_doc_freq[item] += 1

        total_docs = max(len(texts), 1)
        semantic_items = [
            item for item, _ in semantic_counts.most_common(self.u_anchor_count)
        ]
        relation_items = [
            item for item, _ in relation_counts.most_common(self.v_anchor_count)
        ]
        self.u_index = {item: idx for idx, item in enumerate(semantic_items)}
        self.v_index = {item: idx for idx, item in enumerate(relation_items)}
        self.topic_idf = {
            item: math.log((1.0 + total_docs) / (1.0 + semantic_doc_freq[item])) + 1.0
            for item in semantic_items
        }
        self.relation_idf = {
            item: math.log((1.0 + total_docs) / (1.0 + relation_doc_freq[item])) + 1.0
            for item in relation_items
        }
        lexical_df = np.zeros(self.lexical_dim, dtype=np.int32)
        for text in texts:
            seen = set()
            toks = tokenize(text)
            for feat in toks + [f"{a}__{b}" for a, b in zip(toks[:-1], toks[1:])]:
                seen.add(stable_hash(feat, self.lexical_dim))
            for idx in seen:
                lexical_df[idx] += 1
        self.lexical_idf = (
            np.log((1.0 + total_docs) / (1.0 + lexical_df)) + 1.0
        ).astype(np.float32)

    def encode_query(self, text: str) -> QueryRepresentation:
        events = self.text_to_events(text)
        volume = self.events_to_volume(events)
        return QueryRepresentation(
            text=text,
            events=events,
            volume=volume,
            volume_vector=self._volume_vector(volume),
            support_vector=self._support_vector(volume > 0.08 * max(float(volume.max()), 1e-6)),
            time_vector=self._time_vector(volume),
            lexical_vector=self._lexical_vector(text),
            shadow_vector=self._shadow_vector(text),
            semantic_vector=self._semantic_key_vector(events),
            relation_vector=self._relation_key_vector(events),
            semantic_anchor_ratio=self._anchor_ratio(
                [event.semantic_key for event in events],
                self.u_index,
            ),
            relation_anchor_ratio=self._anchor_ratio(
                [event.relation_key for event in events],
                self.v_index,
            ),
            centers=[(event.t_idx, event.u_idx, event.v_idx) for event in events],
        )

    def text_to_events(self, text: str) -> List[SemanticEvent]:
        chunks = sentence_chunks(text, chunk_size=self.chunk_sentences)
        if not chunks:
            return []
        events: List[SemanticEvent] = []
        max_chunk_idx = max(len(chunks) - 1, 1)
        for chunk_idx, chunk in enumerate(chunks):
            entities, topics, flags, emotions = self._extract_semantics(chunk)
            predicates = self._extract_predicates(chunk, flags)
            semantic_units = entities + topics
            if not semantic_units:
                semantic_units = ["misc"]
            if not predicates:
                predicates = list(emotions[:1]) or ["context"]

            t_idx = min(
                int(round(chunk_idx * (self.time_bins - 1) / max_chunk_idx)),
                self.time_bins - 1,
            )
            primary_semantics = semantic_units[:2]
            primary_relations = predicates[:2]
            for semantic_key in primary_semantics:
                for relation_key in primary_relations:
                    weight = self._event_weight(semantic_key, relation_key, flags, chunk)
                    events.append(
                        SemanticEvent(
                            t_idx=t_idx,
                            u_idx=self._u_coord(semantic_key),
                            v_idx=self._v_coord(relation_key),
                            weight=weight,
                            semantic_key=semantic_key,
                            relation_key=relation_key,
                            chunk_index=chunk_idx,
                            chunk_text=chunk,
                            flags=tuple(flags),
                        )
                    )
        return events

    def events_to_volume(self, events: Sequence[SemanticEvent]) -> np.ndarray:
        volume = np.zeros(self.shape, dtype=np.float32)
        if not events:
            return volume

        by_semantic: Dict[str, List[SemanticEvent]] = defaultdict(list)
        by_chunk: Dict[int, List[SemanticEvent]] = defaultdict(list)
        for event in events:
            by_semantic[event.semantic_key].append(event)
            by_chunk[event.chunk_index].append(event)
            self._deposit_kernel(volume, event.t_idx, event.u_idx, event.v_idx, event.weight)

        for semantic_key, semantic_events in by_semantic.items():
            semantic_events = sorted(semantic_events, key=lambda item: item.chunk_index)
            for left, right in zip(semantic_events[:-1], semantic_events[1:]):
                if right.chunk_index - left.chunk_index > 2:
                    continue
                self._draw_connection(volume, left, right, weight=0.55 * min(left.weight, right.weight))

        for chunk_events in by_chunk.values():
            if len(chunk_events) < 2:
                continue
            avg_t = int(round(sum(event.t_idx for event in chunk_events) / len(chunk_events)))
            avg_u = int(round(sum(event.u_idx for event in chunk_events) / len(chunk_events)))
            avg_v = int(round(sum(event.v_idx for event in chunk_events) / len(chunk_events)))
            branch_weight = 0.35 * sum(event.weight for event in chunk_events[:3]) / len(chunk_events[:3])
            self._deposit_kernel(volume, avg_t, avg_u, avg_v, branch_weight)

        max_abs = float(np.max(np.abs(volume)))
        if max_abs > 1e-8:
            volume /= max_abs
        return volume.astype(np.float32)

    def coarse_features(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._volume_vector(volume), self._support_vector(volume > 0.08 * max(float(volume.max()), 1e-6)), self._time_vector(volume)

    def lexical_features(self, text: str) -> np.ndarray:
        return self._lexical_vector(text)

    def shadow_features(self, text: str) -> np.ndarray:
        return self._shadow_vector(text)

    def _extract_semantics(self, chunk: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        compressed = self.dialect.compress(chunk)
        lines = compressed.splitlines()
        content_line = lines[-1] if lines else ""
        parts = content_line.split("|")
        entities_part = parts[0] if len(parts) >= 1 else "0:???"
        topics_part = parts[1] if len(parts) >= 2 else "misc"
        emotions_part = parts[3] if len(parts) >= 4 else ""
        flags_part = parts[4] if len(parts) >= 5 else ""

        entities = []
        if ":" in entities_part:
            entities = [
                item.lower()
                for item in entities_part.split(":", 1)[1].split("+")
                if item and item != "???"
            ]
        topics = [item.lower() for item in topics_part.split("_") if item and item != "misc"]
        flags = [item for item in flags_part.split("+") if item]
        emotions = [item for item in emotions_part.split("+") if item]
        return entities, topics, flags, emotions

    def _extract_predicates(self, chunk: str, flags: Sequence[str]) -> List[str]:
        tokens = tokenize(chunk)
        predicates = []
        for token in tokens:
            if token in STOP_WORDS:
                continue
            if token in COMMON_VERBS or token.endswith(("ed", "ing", "ize", "ise")):
                predicates.append(token)
        for flag in flags:
            mapped = FLAG_TO_RELATION.get(flag)
            if mapped is not None:
                predicates.append(mapped)
        if "?" in chunk:
            predicates.append("question")
        unique = []
        for item in predicates:
            if item not in unique:
                unique.append(item)
        return unique[:4]

    def _u_coord(self, semantic_key: str) -> int:
        if semantic_key in self.u_index:
            return self.u_index[semantic_key]
        if self.u_bins <= self.u_anchor_count:
            return stable_hash(semantic_key, self.u_bins)
        overflow = self.u_bins - self.u_anchor_count
        return self.u_anchor_count + stable_hash(semantic_key, overflow)

    def _v_coord(self, relation_key: str) -> int:
        if relation_key in self.v_index:
            return self.v_index[relation_key]
        if self.v_bins <= self.v_anchor_count:
            return stable_hash(relation_key, self.v_bins)
        overflow = self.v_bins - self.v_anchor_count
        return self.v_anchor_count + stable_hash(relation_key, overflow)

    def _event_weight(
        self,
        semantic_key: str,
        relation_key: str,
        flags: Sequence[str],
        chunk: str,
    ) -> float:
        weight = 1.0
        weight += 0.25 * self.topic_idf.get(semantic_key, 1.0)
        weight += 0.20 * self.relation_idf.get(relation_key, 1.0)
        weight += 0.20 * min(len(flags), 2)
        if any(flag in {"DECISION", "PIVOT", "CORE"} for flag in flags):
            weight += 0.30
        if len(chunk) > 140:
            weight += 0.10
        return float(weight)

    def _deposit_kernel(self, volume: np.ndarray, t_idx: int, u_idx: int, v_idx: int, weight: float) -> None:
        kernel = {
            (0, 0, 0): 1.0,
            (-1, 0, 0): 0.45,
            (1, 0, 0): 0.45,
            (0, -1, 0): 0.45,
            (0, 1, 0): 0.45,
            (0, 0, -1): 0.35,
            (0, 0, 1): 0.35,
        }
        for (dt, du, dv), scale in kernel.items():
            tt = t_idx + dt
            uu = u_idx + du
            vv = v_idx + dv
            if 0 <= tt < self.time_bins and 0 <= uu < self.u_bins and 0 <= vv < self.v_bins:
                volume[tt, uu, vv] += np.float32(weight * scale)

    def _draw_connection(self, volume: np.ndarray, left: SemanticEvent, right: SemanticEvent, weight: float) -> None:
        steps = max(
            abs(right.t_idx - left.t_idx),
            abs(right.u_idx - left.u_idx),
            abs(right.v_idx - left.v_idx),
            1,
        )
        for step in range(steps + 1):
            alpha = step / steps
            t_idx = int(round((1 - alpha) * left.t_idx + alpha * right.t_idx))
            u_idx = int(round((1 - alpha) * left.u_idx + alpha * right.u_idx))
            v_idx = int(round((1 - alpha) * left.v_idx + alpha * right.v_idx))
            self._deposit_kernel(volume, t_idx, u_idx, v_idx, weight)

    def _volume_vector(self, volume: np.ndarray) -> np.ndarray:
        vec = volume.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)

    def _support_vector(self, mask: np.ndarray) -> np.ndarray:
        vec = mask.astype(np.float32).reshape(-1)
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)

    def _time_vector(self, volume: np.ndarray) -> np.ndarray:
        vec = volume.sum(axis=(1, 2)).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)

    def _lexical_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.lexical_dim, dtype=np.float32)
        toks = tokenize(text)
        feats = toks + [f"{a}__{b}" for a, b in zip(toks[:-1], toks[1:])]
        for feat in feats:
            idx = stable_hash(feat, self.lexical_dim)
            vec[idx] += 1.0
        if np.any(vec):
            vec = np.log1p(vec) * self.lexical_idf
            vec /= np.linalg.norm(vec) + 1e-12
        return vec.astype(np.float32)

    def _shadow_vector(self, text: str) -> np.ndarray:
        volume = np.zeros(self.shape, dtype=np.float32)
        toks = tokenize(text)
        if not toks:
            return self._volume_vector(volume)
        chunk_size = max(1, math.ceil(len(toks) / self.time_bins))
        for token_idx, token in enumerate(toks):
            t_idx = min(token_idx // chunk_size, self.time_bins - 1)
            bucket = stable_hash(token, self.u_bins * self.v_bins)
            u_idx = bucket // self.v_bins
            v_idx = bucket % self.v_bins
            volume[t_idx, u_idx, v_idx] += 1.0
        for token_idx, (left, right) in enumerate(zip(toks[:-1], toks[1:])):
            t_idx = min(token_idx // chunk_size, self.time_bins - 1)
            bucket = stable_hash(f"{left}__{right}", self.u_bins * self.v_bins)
            u_idx = bucket // self.v_bins
            v_idx = bucket % self.v_bins
            volume[t_idx, u_idx, v_idx] += 0.5
        return self._volume_vector(volume)

    def _semantic_key_vector(self, events: Sequence[SemanticEvent]) -> np.ndarray:
        weights: Dict[str, float] = defaultdict(float)
        for event in events:
            weights[event.semantic_key] += max(event.weight, 0.25)
        return self._weighted_hash_vector(weights, self.semantic_dim, self.topic_idf)

    def _relation_key_vector(self, events: Sequence[SemanticEvent]) -> np.ndarray:
        weights: Dict[str, float] = defaultdict(float)
        for event in events:
            weights[event.relation_key] += max(event.weight, 0.25)
        return self._weighted_hash_vector(weights, self.relation_dim, self.relation_idf)

    def _weighted_hash_vector(
        self,
        weights: Dict[str, float],
        dim: int,
        idf_lookup: Dict[str, float],
    ) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for key, weight in weights.items():
            idx = stable_hash(key, dim)
            vec[idx] += np.float32(math.sqrt(max(weight, 1e-6)) * idf_lookup.get(key, 1.0))
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)

    def _anchor_ratio(self, keys: Sequence[str], index: Dict[str, int]) -> float:
        if not keys:
            return 0.0
        hits = sum(1 for key in keys if key in index)
        return float(hits / len(keys))


class K3MVBalancedCodec:
    def __init__(
        self,
        shape: Tuple[int, int, int] = (12, 12, 12),
        coarse_factors: Tuple[int, int, int] = (2, 2, 2),
        max_tubes: int = 8,
        max_junctions: int = 4,
        residual_budget: int = 96,
        local_support_radius: int = 1,
    ) -> None:
        self.shape = shape
        self.coarse_factors = coarse_factors
        self.max_tubes = max_tubes
        self.max_junctions = max_junctions
        self.residual_budget = residual_budget
        self.local_support_radius = local_support_radius
        self.base_codec = K3MCodec(
            base_radii=(1, 1, 1),
            tube_half_length=2,
            tube_radius=0,
            max_tubes=max_tubes,
            max_junctions=max_junctions,
            residual_budget=residual_budget,
            response_steps=2,
        )

    def encode(self, volume: np.ndarray) -> K3MV2Archive:
        coarse = average_pool3d(volume, self.coarse_factors)
        coarse_q, coarse_scale = quantize_array(coarse)
        coarse_recon = upsample_nearest(dequantize_array(coarse_q, coarse_scale), volume.shape)
        residual = volume.astype(np.float32) - coarse_recon

        active_direction_indices = self.base_codec._detect_active_directions(residual)
        tubes = self.base_codec._select_tubes(residual, active_direction_indices)
        residual_after_tubes = residual.copy()
        support_count = np.zeros(volume.shape, dtype=np.uint16)
        support_mask = np.zeros(volume.shape, dtype=bool)
        for tube in tubes:
            contribution, mask = self.base_codec._render_tube(volume.shape, tube)
            residual_after_tubes -= contribution
            support_count += mask.astype(np.uint16)
            support_mask |= mask

        junctions = self.base_codec._extract_junctions(residual_after_tubes, support_count, tubes)
        residual_after_junctions = residual_after_tubes.copy()
        for junction in junctions:
            contribution, mask = self.base_codec._render_junction(volume.shape, junction, return_mask=True)
            residual_after_junctions -= contribution
            support_mask |= mask

        sparse_residual = self._encode_sparse_residual_near_support(
            residual_after_junctions,
            support_mask=support_mask,
        )
        if sparse_residual.indices.size:
            support_mask |= self._indices_to_mask(volume.shape, sparse_residual.indices)
        support_ratio = float(np.mean(support_mask))
        metadata = {
            "support_ratio": support_ratio,
            "n_tubes": float(len(tubes)),
            "n_junctions": float(len(junctions)),
            "active_direction_count": float(len(active_direction_indices)),
            "covered_direction_count": float(len({tube.direction_idx for tube in tubes})),
            "coarse_ratio": float(np.prod(coarse.shape) / max(np.prod(volume.shape), 1)),
            "archive_type": "k3m_v2_balanced",
        }
        return K3MV2Archive(
            shape=tuple(volume.shape),
            coarse_shape=tuple(coarse.shape),
            coarse_q=coarse_q,
            coarse_scale=coarse_scale,
            tubes=tubes,
            junctions=junctions,
            residual=sparse_residual,
            direction_bank=self.base_codec.direction_bank,
            metadata=metadata,
        )

    def decode(self, archive: K3MV2Archive) -> np.ndarray:
        coarse = dequantize_array(archive.coarse_q, archive.coarse_scale)
        recon = upsample_nearest(coarse, archive.shape)
        for tube in archive.tubes:
            contribution, _ = self.base_codec._render_tube(archive.shape, tube)
            recon += contribution
        for junction in archive.junctions:
            recon += self.base_codec._render_junction(archive.shape, junction)
        if archive.residual.indices.size:
            values = dequantize_array(archive.residual.q_values, archive.residual.scale)
            idx = archive.residual.indices
            recon[idx[:, 0], idx[:, 1], idx[:, 2]] += values
        return recon.astype(np.float32)

    def support_mask(self, archive: K3MV2Archive) -> np.ndarray:
        mask = np.zeros(archive.shape, dtype=bool)
        for tube in archive.tubes:
            _, tube_mask = self.base_codec._render_tube(archive.shape, tube)
            mask |= tube_mask
        for junction in archive.junctions:
            _, junction_mask = self.base_codec._render_junction(archive.shape, junction, return_mask=True)
            mask |= junction_mask
        if archive.residual.indices.size:
            mask |= self._indices_to_mask(archive.shape, archive.residual.indices)
        return mask

    def decode_local(
        self,
        archive: K3MV2Archive,
        center: Tuple[int, int, int],
        radius: Tuple[int, int, int] = (1, 2, 2),
    ) -> np.ndarray:
        recon = self.decode(archive)
        slices = []
        for axis, value in enumerate(center):
            lo = max(0, value - radius[axis])
            hi = min(archive.shape[axis], value + radius[axis] + 1)
            slices.append(slice(lo, hi))
        return recon[tuple(slices)]

    def _encode_sparse_residual_near_support(
        self,
        residual: np.ndarray,
        support_mask: np.ndarray,
    ) -> SparseResidual:
        candidate_mask = dilate_mask(support_mask, radius=self.local_support_radius)
        if not np.any(candidate_mask):
            candidate_mask = np.abs(residual) > 0
        coords = np.argwhere(candidate_mask)
        if coords.size == 0:
            return SparseResidual(
                indices=np.empty((0, 3), dtype=np.int16),
                q_values=np.empty(0, dtype=np.int16),
                scale=1.0,
            )
        values = residual[candidate_mask]
        budget = min(self.residual_budget, values.size)
        idx = np.argpartition(np.abs(values), -budget)[-budget:]
        idx = idx[np.argsort(-np.abs(values[idx]))]
        chosen_coords = coords[idx].astype(np.int16)
        chosen_values = values[idx].astype(np.float32)
        q_values, scale = quantize_array(chosen_values)
        return SparseResidual(indices=chosen_coords, q_values=q_values, scale=scale)

    def _indices_to_mask(self, shape: Tuple[int, int, int], indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        if indices.size:
            mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        return mask


def build_v2_query(projector: SemanticTextProjectorV2, text: str) -> QueryRepresentation:
    return projector.encode_query(text)


def build_v2_doc_features(
    projector: SemanticTextProjectorV2,
    codec: K3MVBalancedCodec,
    archive: K3MV2Archive,
    original_text: str,
    doc_repr: Optional[QueryRepresentation] = None,
) -> Dict[str, object]:
    recon = codec.decode(archive)
    support_mask = codec.support_mask(archive)
    volume_vector, support_vector, time_vector = projector.coarse_features(recon)
    if doc_repr is None:
        doc_repr = projector.encode_query(original_text)
    return {
        "recon": recon,
        "support_mask": support_mask,
        "volume_vector": volume_vector,
        "support_vector": support_vector,
        "time_vector": time_vector,
        "lexical_vector": projector.lexical_features(original_text),
        "shadow_vector": projector.shadow_features(original_text),
        "semantic_vector": doc_repr.semantic_vector,
        "relation_vector": doc_repr.relation_vector,
        "semantic_anchor_ratio": doc_repr.semantic_anchor_ratio,
        "relation_anchor_ratio": doc_repr.relation_anchor_ratio,
    }


def score_v2_candidate(
    query: QueryRepresentation,
    doc_features: Dict[str, object],
    local_weight: float = 0.02,
) -> float:
    coarse_mix_score = doc_features.get("coarse_mix_score")
    if coarse_mix_score is None:
        lexical_similarity = cosine_similarity(query.lexical_vector, doc_features["lexical_vector"])
        semantic_similarity = cosine_similarity(query.semantic_vector, doc_features["semantic_vector"])
        relation_similarity = cosine_similarity(query.relation_vector, doc_features["relation_vector"])
        shadow_similarity = cosine_similarity(query.shadow_vector, doc_features["shadow_vector"])
        structure_similarity = (
            0.70 * cosine_similarity(query.volume_vector, doc_features["volume_vector"])
            + 0.20 * cosine_similarity(query.support_vector, doc_features["support_vector"])
            + 0.10 * cosine_similarity(query.time_vector, doc_features["time_vector"])
        )
        coarse_mix_score = (
            0.42 * lexical_similarity
            + 0.18 * semantic_similarity
            + 0.05 * relation_similarity
            + 0.15 * shadow_similarity
            + 0.20 * structure_similarity
        )
    return float(float(coarse_mix_score) + local_weight * float(doc_features.get("local_match", 0.0)))


def compute_local_match(
    query: QueryRepresentation,
    codec: K3MVBalancedCodec,
    archive: K3MV2Archive,
) -> float:
    if not query.centers:
        return 0.0
    local_scores = []
    for center in query.centers[:4]:
        patch = codec.decode_local(archive, center)
        query_patch = extract_local_patch(query.volume, center)
        if patch.size == 0 or query_patch.size == 0:
            continue
        patch_vec = patch.reshape(-1).astype(np.float32)
        query_vec = query_patch.reshape(-1).astype(np.float32)
        local_scores.append(cosine_similarity(query_vec, patch_vec))
    if not local_scores:
        return 0.0
    return float(np.mean(local_scores))


def extract_local_patch(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    radius: Tuple[int, int, int] = (1, 2, 2),
) -> np.ndarray:
    spans = []
    for axis, value in enumerate(center):
        lo = max(0, value - radius[axis])
        hi = min(volume.shape[axis], value + radius[axis] + 1)
        spans.append(slice(lo, hi))
    return volume[tuple(spans)]


def build_v2_cache(
    texts_by_id: Dict[str, str],
    shape: Tuple[int, int, int] = (12, 12, 12),
    coarse_factors: Tuple[int, int, int] = (2, 2, 2),
    max_tubes: int = 8,
    max_junctions: int = 4,
    residual_budget: int = 96,
) -> Tuple[SemanticTextProjectorV2, K3MVBalancedCodec, Dict[str, Dict[str, object]], Dict[str, float]]:
    projector = SemanticTextProjectorV2(shape=shape)
    projector.fit(list(texts_by_id.values()))
    codec = K3MVBalancedCodec(
        shape=shape,
        coarse_factors=coarse_factors,
        max_tubes=max_tubes,
        max_junctions=max_junctions,
        residual_budget=residual_budget,
    )
    cache: Dict[str, Dict[str, object]] = {}
    encode_times = []
    decode_times = []
    archive_sizes = []
    raw_sizes = []

    for idx, (session_id, text) in enumerate(texts_by_id.items(), start=1):
        query = projector.encode_query(text)
        start_encode = time.perf_counter()
        archive = codec.encode(query.volume)
        end_encode = time.perf_counter()
        features = build_v2_doc_features(
            projector,
            codec,
            archive,
            original_text=text,
            doc_repr=query,
        )
        end_decode = time.perf_counter()
        cache[session_id] = {
            "archive": archive,
            "query": query,
            **features,
            "archive_bytes": archive.compressed_size_bytes(),
            "raw_bytes": len(text.encode("utf-8")),
        }
        encode_times.append(end_encode - start_encode)
        decode_times.append(end_decode - end_encode)
        archive_sizes.append(archive.compressed_size_bytes())
        raw_sizes.append(len(text.encode("utf-8")))
        if idx % 2000 == 0:
            print(f"[k3m-v2-cache] encoded {idx}/{len(texts_by_id)} sessions")

    raw_arr = np.array(raw_sizes, dtype=np.float64)
    archive_arr = np.array(archive_sizes, dtype=np.float64)
    stats = {
        "avg_raw_bytes": float(raw_arr.mean()),
        "avg_index_bytes": float(archive_arr.mean()),
        "avg_compression_ratio": float((raw_arr / np.maximum(archive_arr, 1.0)).mean()),
        "avg_encode_seconds": float(np.mean(encode_times) if encode_times else 0.0),
        "avg_decode_seconds": float(np.mean(decode_times) if decode_times else 0.0),
        "avg_support_ratio": float(
            np.mean([item["archive"].metadata.get("support_ratio", 0.0) for item in cache.values()])
            if cache
            else 0.0
        ),
    }
    return projector, codec, cache, stats


def rank_v2_candidates(
    query: QueryRepresentation,
    candidate_ids: Sequence[str],
    cache: Dict[str, Dict[str, object]],
    codec: K3MVBalancedCodec,
    refine_top_n: int = 24,
) -> List[str]:
    coarse_scores = []
    for session_id in candidate_ids:
        doc = cache[session_id]
        lexical_score = cosine_similarity(query.lexical_vector, doc["lexical_vector"])
        semantic_score = (
            0.70 * cosine_similarity(query.semantic_vector, doc["semantic_vector"])
            + 0.30 * cosine_similarity(query.relation_vector, doc["relation_vector"])
        )
        shadow_score = cosine_similarity(query.shadow_vector, doc["shadow_vector"])
        structure_score = (
            0.70 * cosine_similarity(query.volume_vector, doc["volume_vector"])
            + 0.20 * cosine_similarity(query.support_vector, doc["support_vector"])
            + 0.10 * cosine_similarity(query.time_vector, doc["time_vector"])
        )
        mix_score = (
            0.45 * lexical_score
            + 0.23 * semantic_score
            + 0.12 * shadow_score
            + 0.20 * structure_score
        )
        doc["coarse_mix_score"] = float(mix_score)
        coarse_scores.append(
            {
                "session_id": session_id,
                "mix_score": float(mix_score),
                "lexical_score": float(lexical_score),
                "semantic_score": float(semantic_score),
                "shadow_score": float(shadow_score),
            }
        )
    mix_sorted = sorted(coarse_scores, key=lambda item: item["mix_score"], reverse=True)

    shortlist_ids: List[str] = []
    shortlist_seen = set()
    shortlist_specs = (
        ("mix_score", refine_top_n),
        ("lexical_score", refine_top_n),
        ("semantic_score", max(8, refine_top_n // 2)),
        ("shadow_score", max(8, refine_top_n // 2)),
    )
    for score_key, limit in shortlist_specs:
        ranked = sorted(coarse_scores, key=lambda item: item[score_key], reverse=True)
        for item in ranked[: min(limit, len(ranked))]:
            session_id = item["session_id"]
            if session_id in shortlist_seen:
                continue
            shortlist_seen.add(session_id)
            shortlist_ids.append(session_id)

    refined = []
    for session_id in shortlist_ids:
        local_match = compute_local_match(query, codec, cache[session_id]["archive"])
        cache[session_id]["local_match"] = local_match
        refined.append(
            (
                score_v2_candidate(query, cache[session_id], local_weight=0.02),
                session_id,
            )
        )
    refined.sort(key=lambda item: item[0], reverse=True)
    refined_ids = [session_id for _, session_id in refined]
    coarse_tail = [
        item["session_id"]
        for item in mix_sorted
        if item["session_id"] not in refined_ids
    ]
    return refined_ids + coarse_tail
