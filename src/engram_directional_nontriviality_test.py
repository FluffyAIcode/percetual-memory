from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from benchmark_compare_k3m_aaak import build_session_catalog, load_longmemeval
from constraint_edge_volume import ConstraintEdgeRecord, ConstraintEdgeVolumeProjector
from k3m_codec import K3MCodec
from k3m_v2_balanced import SemanticTextProjectorV2, sentence_chunks, stable_hash, tokenize
from llm_state_extractor import (
    LLMStateExtractor,
    llm_chunk_states_to_state_fields,
    llm_constraint_steps_to_state_fields,
)


STATE_BASED_MODES = {
    "relation-aware",
    "state-address",
    "canonical-state-address",
    "semi-canonical-state-address",
    "llm-state-address",
    "llm-rotating-phase-state-address",
    "llm-event-phase-state-address",
    "llm-event-transition-address",
    "llm-finite-event-transition-address",
    "llm-constraint-object-address",
    "llm-constraint-edge-address",
    "llm-constraint-edge-volume",
}

CAUSAL_MARKERS = {
    "because": "cause",
    "since": "cause",
    "therefore": "effect",
    "thus": "effect",
    "hence": "effect",
    "so": "effect",
    "result": "effect",
    "caused": "cause",
    "causes": "cause",
    "led": "effect",
    "due": "cause",
    "if": "condition",
    "unless": "condition",
}

TURN_MARKERS = {
    "but": "contrast",
    "however": "contrast",
    "although": "contrast",
    "though": "contrast",
    "instead": "redirect",
    "rather": "redirect",
    "yet": "contrast",
    "then": "progression",
    "finally": "resolution",
    "eventually": "resolution",
    "meanwhile": "parallel",
    "before": "setup",
    "after": "aftermath",
    "suddenly": "surprise",
    "question": "inquiry",
}


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + 1e-12
    return (vec / norm).astype(np.float32)


def _sigmoid(value: float) -> float:
    value = max(min(value, 20.0), -20.0)
    return float(1.0 / (1.0 + math.exp(-value)))


def _entropy(prob: np.ndarray) -> float:
    valid = prob[prob > 1e-12]
    if valid.size == 0:
        return 0.0
    return float(-np.sum(valid * np.log(valid)))


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 1e-12) & (q > 1e-12)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (_kl_div(p, m) + _kl_div(q, m))


def _top_mass(prob: np.ndarray, k: int) -> float:
    if prob.size == 0:
        return 0.0
    idx = np.argsort(-prob)[:k]
    return float(np.sum(prob[idx]))


def _top_items(weights: Dict[int, float], limit: int) -> Dict[int, float]:
    if limit <= 0 or len(weights) <= limit:
        return dict(weights)
    ranked = sorted(weights.items(), key=lambda item: -item[1])[:limit]
    return {idx: float(value) for idx, value in ranked}


def _normalize_counter(counter: Counter[str]) -> np.ndarray:
    values = np.array(list(counter.values()), dtype=np.float64)
    total = float(np.sum(values))
    if total <= 1e-12 or values.size == 0:
        return np.zeros(0, dtype=np.float64)
    return values / total


def _gini(weights: Sequence[float]) -> float:
    arr = np.array([max(float(item), 0.0) for item in weights], dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 1e-12 or arr.size == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * arr) / (n * total)) - (n + 1.0) / n)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize_complex(vec: np.ndarray) -> np.ndarray:
    norm = float(np.sqrt(np.sum(np.abs(vec) ** 2))) + 1e-12
    return (vec / norm).astype(np.complex64)


def _complex_alignment(a: np.ndarray, b: np.ndarray) -> complex:
    denom = float(np.sqrt(np.sum(np.abs(a) ** 2)) * np.sqrt(np.sum(np.abs(b) ** 2))) + 1e-12
    if denom <= 1e-12:
        return 0.0 + 0.0j
    return complex(np.vdot(a, b) / denom)


def shuffled_chunk_text(text: str, seed: int) -> str:
    chunks = sentence_chunks(text, chunk_size=2)
    if len(chunks) <= 1:
        return text
    rng = random.Random(seed)
    shuffled = chunks[:]
    rng.shuffle(shuffled)
    return ". ".join(shuffled)


@dataclass
class ChunkActivation:
    address_weights: Dict[int, float]
    gate_values: Dict[int, float]
    chunk_text: str


@dataclass
class ChunkState:
    chunk_text: str
    tokens: List[str]
    semantic_keys: List[str]
    relation_keys: List[str]
    flags: List[str]
    emotions: List[str]
    causal_markers: List[str]
    turn_markers: List[str]
    event_roles: List[str]
    causal_schemas: List[str]
    narrative_acts: List[str]
    weight: float


@dataclass
class EventTransitionState:
    phase_bucket: int
    strength: float
    event_tags: List[str]
    tone_shift: str
    topic_shift: str
    causal_shift: str
    relation_signature: str
    narrative_signature: str
    semantic_anchor: str


class ObservableEngram:
    def __init__(
        self,
        address_count: int = 4096,
        context_dim: int = 256,
        ngram_orders: Sequence[int] = (2, 3),
        hash_heads: int = 4,
        chunk_sentences: int = 2,
    ) -> None:
        self.address_count = int(address_count)
        self.context_dim = int(context_dim)
        self.ngram_orders = tuple(int(order) for order in ngram_orders)
        self.hash_heads = int(hash_heads)
        self.chunk_sentences = int(chunk_sentences)
        self.idf: Dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> "ObservableEngram":
        doc_freq: Counter[str] = Counter()
        for text in texts:
            seen = set(self._context_features(text))
            doc_freq.update(seen)
        total_docs = max(len(texts), 1)
        self.idf = {
            feat: float(math.log((1.0 + total_docs) / (1.0 + freq)) + 1.0)
            for feat, freq in doc_freq.items()
        }
        return self

    def encode_text(self, text: str) -> List[ChunkActivation]:
        chunks = sentence_chunks(text, chunk_size=self.chunk_sentences)
        if not chunks:
            return []

        activations: List[ChunkActivation] = []
        prefix_tokens: List[str] = []
        for chunk in chunks:
            chunk_tokens = tokenize(chunk)
            if not chunk_tokens:
                continue
            prefix_tokens.extend(chunk_tokens)
            context_vec = self._context_vector(prefix_tokens[-64:])
            address_features = self._address_features(chunk_tokens)
            if not address_features:
                continue

            address_weights: Dict[int, float] = defaultdict(float)
            gate_values: Dict[int, float] = {}
            for address_id, feature_weight in address_features.items():
                address_vec = self._address_vector(address_id)
                score = _cosine_similarity(context_vec, address_vec)
                gate = _sigmoid(2.5 * score)
                address_weights[address_id] += float(feature_weight * gate)
                gate_values[address_id] = gate
            activations.append(
                ChunkActivation(
                    address_weights=dict(address_weights),
                    gate_values=gate_values,
                    chunk_text=chunk,
                )
            )
        return activations

    def _context_features(self, text: str) -> List[str]:
        toks = tokenize(text)
        if not toks:
            return []
        features = list(toks)
        for order in self.ngram_orders:
            if len(toks) < order:
                continue
            for start in range(len(toks) - order + 1):
                features.append("__".join(toks[start : start + order]))
        return features

    def _context_vector(self, tokens: Sequence[str]) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.float32)
        if not tokens:
            return vec
        weights: Dict[str, float] = defaultdict(float)
        token_list = list(tokens)
        for token in token_list:
            weights[token] += self.idf.get(token, 1.0)
        for order in self.ngram_orders:
            if len(token_list) < order:
                continue
            for start in range(len(token_list) - order + 1):
                feat = "__".join(token_list[start : start + order])
                weights[feat] += 1.25 * self.idf.get(feat, 1.0)
        for feat, weight in weights.items():
            idx = stable_hash(feat, self.context_dim)
            sign = -1.0 if stable_hash(f"sgn::{feat}", 2) == 0 else 1.0
            vec[idx] += np.float32(weight * sign)
        return _normalize(vec)

    def _address_features(self, tokens: Sequence[str]) -> Dict[int, float]:
        feats: Dict[int, float] = defaultdict(float)
        counts = Counter(tokens)
        for token, count in counts.items():
            base_weight = math.sqrt(float(count)) * self.idf.get(token, 1.0)
            for head in range(self.hash_heads):
                address_id = stable_hash(f"1::{head}::{token}", self.address_count)
                feats[address_id] += float(base_weight / self.hash_heads)

        token_list = list(tokens)
        for order in self.ngram_orders:
            if len(token_list) < order:
                continue
            ngrams = ["__".join(token_list[start : start + order]) for start in range(len(token_list) - order + 1)]
            ngram_counts = Counter(ngrams)
            for feat, count in ngram_counts.items():
                base_weight = (1.2 + 0.2 * order) * math.sqrt(float(count)) * self.idf.get(feat, 1.0)
                for head in range(self.hash_heads):
                    address_id = stable_hash(f"{order}::{head}::{feat}", self.address_count)
                    feats[address_id] += float(base_weight / self.hash_heads)
        return feats

    def _address_vector(self, address_id: int) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.float32)
        for lane in range(4):
            idx = stable_hash(f"{address_id}::lane::{lane}", self.context_dim)
            sign = -1.0 if stable_hash(f"{address_id}::sign::{lane}", 2) == 0 else 1.0
            vec[idx] += np.float32(sign)
        return _normalize(vec)


class RelationAwareObservableEngram:
    def __init__(
        self,
        address_count: int = 4096,
        context_dim: int = 256,
        ngram_orders: Sequence[int] = (2, 3),
        hash_heads: int = 4,
        chunk_sentences: int = 2,
        projector_shape: Tuple[int, int, int] = (12, 12, 12),
    ) -> None:
        self.address_count = int(address_count)
        self.context_dim = int(context_dim)
        self.ngram_orders = tuple(int(order) for order in ngram_orders)
        self.hash_heads = int(hash_heads)
        self.chunk_sentences = int(chunk_sentences)
        self.projector = SemanticTextProjectorV2(shape=projector_shape, chunk_sentences=chunk_sentences)
        self.idf: Dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> "RelationAwareObservableEngram":
        self.projector.fit(texts)
        doc_freq: Counter[str] = Counter()
        for text in texts:
            seen = set(self._all_features_from_states(self.extract_states(text)))
            doc_freq.update(seen)
        total_docs = max(len(texts), 1)
        self.idf = {
            feat: float(math.log((1.0 + total_docs) / (1.0 + freq)) + 1.0)
            for feat, freq in doc_freq.items()
        }
        return self

    def extract_states(self, text: str) -> List[ChunkState]:
        chunks = sentence_chunks(text, chunk_size=self.chunk_sentences)
        if not chunks:
            return []
        states: List[ChunkState] = []
        total_chunks = len(chunks)
        for chunk_idx, chunk in enumerate(chunks):
            tokens = tokenize(chunk)
            entities, topics, flags, emotions = self.projector._extract_semantics(chunk)
            relations = self.projector._extract_predicates(chunk, flags)
            semantic_keys = list(dict.fromkeys((entities + topics)[:4])) or ["misc"]
            relation_keys = list(dict.fromkeys(relations[:4])) or ["context"]
            causal_markers = self._extract_causal_markers(chunk, tokens, relation_keys)
            turn_markers = self._extract_turn_markers(chunk, tokens, flags)
            event_roles = self._extract_event_roles(
                semantic_keys=semantic_keys,
                relation_keys=relation_keys,
                causal_markers=causal_markers,
                turn_markers=turn_markers,
                flags=flags,
            )
            causal_schemas = self._extract_causal_schemas(
                relation_keys=relation_keys,
                causal_markers=causal_markers,
                flags=flags,
            )
            narrative_acts = self._extract_narrative_acts(
                chunk_idx=chunk_idx,
                total_chunks=total_chunks,
                flags=flags,
                turn_markers=turn_markers,
                relation_keys=relation_keys,
            )
            weight = 1.0 + 0.15 * len(semantic_keys) + 0.10 * len(relation_keys)
            weight += 0.08 * len(causal_markers) + 0.08 * len(turn_markers)
            weight += 0.08 * len(event_roles) + 0.10 * len(causal_schemas) + 0.10 * len(narrative_acts)
            if any(flag in {"DECISION", "PIVOT", "CORE"} for flag in flags):
                weight += 0.25
            states.append(
                ChunkState(
                    chunk_text=chunk,
                    tokens=tokens,
                    semantic_keys=semantic_keys,
                    relation_keys=relation_keys,
                    flags=list(flags),
                    emotions=list(emotions),
                    causal_markers=causal_markers,
                    turn_markers=turn_markers,
                    event_roles=event_roles,
                    causal_schemas=causal_schemas,
                    narrative_acts=narrative_acts,
                    weight=float(weight),
                )
            )
        return states

    def _extract_event_roles(
        self,
        semantic_keys: Sequence[str],
        relation_keys: Sequence[str],
        causal_markers: Sequence[str],
        turn_markers: Sequence[str],
        flags: Sequence[str],
    ) -> List[str]:
        roles: List[str] = []
        if semantic_keys:
            roles.append(f"actor::{semantic_keys[0]}")
        if len(semantic_keys) >= 2:
            roles.append(f"target::{semantic_keys[1]}")
        if relation_keys:
            roles.append(f"action::{relation_keys[0]}")
        if len(relation_keys) >= 2:
            roles.append(f"modifier::{relation_keys[1]}")
        if causal_markers:
            roles.append(f"motive::{causal_markers[0]}")
        if turn_markers:
            roles.append(f"turn::{turn_markers[0]}")
        if "DECISION" in flags:
            roles.append("outcome::decision")
        if "PIVOT" in flags:
            roles.append("outcome::pivot")
        if "CORE" in flags:
            roles.append("outcome::principle")
        return list(dict.fromkeys(roles))[:8]

    def _extract_causal_schemas(
        self,
        relation_keys: Sequence[str],
        causal_markers: Sequence[str],
        flags: Sequence[str],
    ) -> List[str]:
        schemas: List[str] = []
        markers = set(causal_markers)
        relations = set(relation_keys)
        if "cause" in markers and ("decision" in relations or "DECISION" in flags):
            schemas.append("schema::cause_to_decision")
        if "effect" in markers and ("transition" in relations or "PIVOT" in flags):
            schemas.append("schema::decision_to_effect")
        if "condition" in markers:
            schemas.append("schema::conditional_branch")
        if "deliberation" in markers:
            schemas.append("schema::deliberation_loop")
        if "uncertainty" in markers or "question" in relations:
            schemas.append("schema::uncertain_resolution")
        if "origin" in relations or "GENESIS" in flags or "ORIGIN" in flags:
            schemas.append("schema::origin_chain")
        return list(dict.fromkeys(schemas))[:6]

    def _extract_narrative_acts(
        self,
        chunk_idx: int,
        total_chunks: int,
        flags: Sequence[str],
        turn_markers: Sequence[str],
        relation_keys: Sequence[str],
    ) -> List[str]:
        acts: List[str] = []
        if chunk_idx == 0:
            acts.append("act::opening")
        elif chunk_idx == total_chunks - 1:
            acts.append("act::closing")
        else:
            acts.append("act::middle")
        if "origin" in turn_markers or "ORIGIN" in flags or "GENESIS" in flags:
            acts.append("act::setup")
        if "contrast" in turn_markers or "redirect" in turn_markers:
            acts.append("act::complication")
        if "PIVOT" in flags or "pivot" in turn_markers:
            acts.append("act::turn")
        if "DECISION" in flags or "commit" in turn_markers:
            acts.append("act::commitment")
        if "resolution" in turn_markers or (chunk_idx >= max(total_chunks - 2, 0) and "decision" in relation_keys):
            acts.append("act::resolution")
        if "question" in relation_keys or "inquiry" in turn_markers:
            acts.append("act::inquiry")
        return list(dict.fromkeys(acts))[:6]

    def _extract_causal_markers(self, chunk: str, tokens: Sequence[str], relation_keys: Sequence[str]) -> List[str]:
        found: List[str] = []
        lowered = chunk.lower()
        for marker, label in CAUSAL_MARKERS.items():
            if marker in lowered or marker in tokens:
                found.append(label)
        if "decision" in relation_keys and "cause" not in found:
            found.append("deliberation")
        if "question" in relation_keys and "condition" not in found:
            found.append("uncertainty")
        return list(dict.fromkeys(found))[:4]

    def _extract_turn_markers(self, chunk: str, tokens: Sequence[str], flags: Sequence[str]) -> List[str]:
        found: List[str] = []
        lowered = chunk.lower()
        for marker, label in TURN_MARKERS.items():
            if marker == "question":
                if "?" in chunk:
                    found.append(label)
                continue
            if marker in lowered or marker in tokens:
                found.append(label)
        if "PIVOT" in flags:
            found.append("pivot")
        if "DECISION" in flags:
            found.append("commit")
        if "GENESIS" in flags or "ORIGIN" in flags:
            found.append("origin")
        return list(dict.fromkeys(found))[:4]

    def encode_states(self, states: Sequence[ChunkState]) -> List[ChunkActivation]:
        if not states:
            return []
        activations: List[ChunkActivation] = []
        prefix_states: List[ChunkState] = []
        previous_top_addresses: List[int] = []
        for state in states:
            prefix_states.append(state)
            address_features = self._address_features(state, previous_top_addresses)
            if not address_features:
                continue
            context_vec = self._context_vector(prefix_states)
            address_weights: Dict[int, float] = defaultdict(float)
            gate_values: Dict[int, float] = {}
            relation_boost = 1.0 + 0.08 * len(state.relation_keys)
            if any(flag in {"DECISION", "PIVOT"} for flag in state.flags):
                relation_boost += 0.12
            for address_id, feature_weight in address_features.items():
                address_vec = self._address_vector(address_id)
                score = _cosine_similarity(context_vec, address_vec)
                gate = _sigmoid(3.0 * score)
                address_weights[address_id] += float(feature_weight * gate * relation_boost)
                gate_values[address_id] = gate
            trimmed = _top_items(dict(address_weights), 24)
            previous_top_addresses = list(trimmed.keys())[:8]
            activations.append(
                ChunkActivation(
                    address_weights=trimmed,
                    gate_values={addr: gate_values[addr] for addr in trimmed},
                    chunk_text=state.chunk_text,
                )
            )
        return activations

    def encode_text(self, text: str) -> List[ChunkActivation]:
        return self.encode_states(self.extract_states(text))

    def _all_features_from_states(self, states: Sequence[ChunkState]) -> List[str]:
        features: List[str] = []
        for state in states:
            features.extend(self._state_features(state))
        return features

    def _state_features(self, state: ChunkState) -> List[str]:
        features: List[str] = []
        features.extend(state.tokens)
        for order in self.ngram_orders:
            if len(state.tokens) < order:
                continue
            for start in range(len(state.tokens) - order + 1):
                features.append("__".join(state.tokens[start : start + order]))
        features.extend([f"sem::{item}" for item in state.semantic_keys])
        features.extend([f"rel::{item}" for item in state.relation_keys])
        features.extend([f"flag::{item}" for item in state.flags])
        features.extend([f"emo::{item}" for item in state.emotions])
        features.extend([f"cause::{item}" for item in state.causal_markers])
        features.extend([f"turn::{item}" for item in state.turn_markers])
        features.extend([f"role::{item}" for item in state.event_roles])
        features.extend([f"schema::{item}" for item in state.causal_schemas])
        features.extend([f"act::{item}" for item in state.narrative_acts])
        features.extend(
            [
                f"pair::{sem}::{rel}"
                for sem in state.semantic_keys[:3]
                for rel in state.relation_keys[:3]
            ]
        )
        return features

    def _context_vector(self, prefix_states: Sequence[ChunkState]) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.float32)
        if not prefix_states:
            return vec
        weights: Dict[str, float] = defaultdict(float)
        recent_states = list(prefix_states[-8:])
        for offset, state in enumerate(recent_states):
            decay = 1.0 + 0.35 * offset / max(len(recent_states), 1)
            for feat in self._state_features(state):
                weights[feat] += decay * self.idf.get(feat, 1.0)
            for sem in state.semantic_keys[:3]:
                for rel in state.relation_keys[:3]:
                    bridge = f"bridge::{sem}=>{rel}"
                    weights[bridge] += 1.4 * decay * self.idf.get(bridge, 1.0)
        for left, right in zip(recent_states[:-1], recent_states[1:]):
            for src_rel in left.relation_keys[:2]:
                for dst_rel in right.relation_keys[:2]:
                    feat = f"relflow::{src_rel}=>{dst_rel}"
                    weights[feat] += 1.6 * self.idf.get(feat, 1.0)
            for src_sem in left.semantic_keys[:2]:
                for dst_sem in right.semantic_keys[:2]:
                    feat = f"semflow::{src_sem}=>{dst_sem}"
                    weights[feat] += 1.3 * self.idf.get(feat, 1.0)
        for feat, weight in weights.items():
            idx = stable_hash(feat, self.context_dim)
            sign = -1.0 if stable_hash(f"sgn::{feat}", 2) == 0 else 1.0
            vec[idx] += np.float32(weight * sign)
        return _normalize(vec)

    def _address_features(self, state: ChunkState, previous_top_addresses: Sequence[int]) -> Dict[int, float]:
        feats: Dict[int, float] = defaultdict(float)
        token_counts = Counter(state.tokens)
        for token, count in token_counts.items():
            base_weight = math.sqrt(float(count)) * self.idf.get(token, 1.0)
            for head in range(self.hash_heads):
                address_id = stable_hash(f"tok::{head}::{token}", self.address_count)
                feats[address_id] += float(base_weight / self.hash_heads)

        for order in self.ngram_orders:
            if len(state.tokens) < order:
                continue
            ngrams = ["__".join(state.tokens[start : start + order]) for start in range(len(state.tokens) - order + 1)]
            ngram_counts = Counter(ngrams)
            for feat, count in ngram_counts.items():
                base_weight = (1.1 + 0.2 * order) * math.sqrt(float(count)) * self.idf.get(feat, 1.0)
                for head in range(self.hash_heads):
                    address_id = stable_hash(f"ng::{order}::{head}::{feat}", self.address_count)
                    feats[address_id] += float(base_weight / self.hash_heads)

        for sem in state.semantic_keys[:4]:
            base_weight = 1.7 * self.idf.get(f"sem::{sem}", self.idf.get(sem, 1.0))
            for head in range(self.hash_heads):
                feats[stable_hash(f"sem::{head}::{sem}", self.address_count)] += float(base_weight / self.hash_heads)

        for rel in state.relation_keys[:4]:
            base_weight = 1.8 * self.idf.get(f"rel::{rel}", self.idf.get(rel, 1.0))
            for head in range(self.hash_heads):
                feats[stable_hash(f"rel::{head}::{rel}", self.address_count)] += float(base_weight / self.hash_heads)

        for sem in state.semantic_keys[:3]:
            for rel in state.relation_keys[:3]:
                feat = f"pair::{sem}::{rel}"
                base_weight = 2.2 * self.idf.get(feat, 1.0) * state.weight
                for head in range(self.hash_heads):
                    feats[stable_hash(f"pair::{head}::{feat}", self.address_count)] += float(base_weight / self.hash_heads)

        for flag in state.flags[:3]:
            feat = f"flag::{flag}"
            base_weight = 1.6 * self.idf.get(feat, 1.0)
            feats[stable_hash(f"flag::{feat}", self.address_count)] += float(base_weight)

        for prev_address in previous_top_addresses[:8]:
            for rel in state.relation_keys[:2]:
                feat = f"carry::{prev_address}::{rel}"
                feats[stable_hash(feat, self.address_count)] += float(0.9 * self.idf.get(f"rel::{rel}", 1.0))
        return feats

    def _address_vector(self, address_id: int) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.float32)
        for lane in range(6):
            idx = stable_hash(f"{address_id}::lane::{lane}", self.context_dim)
            sign = -1.0 if stable_hash(f"{address_id}::sign::{lane}", 2) == 0 else 1.0
            vec[idx] += np.float32(sign)
        return _normalize(vec)


class StateAddressEngram(RelationAwareObservableEngram):
    def encode_states(self, states: Sequence[ChunkState]) -> List[ChunkActivation]:
        if not states:
            return []
        activations: List[ChunkActivation] = []
        prefix_states: List[ChunkState] = []
        previous_state: ChunkState | None = None
        previous_top_addresses: List[int] = []
        for state in states:
            prefix_states.append(state)
            context_vec = self._state_context_vector(prefix_states)
            feature_items = self._address_feature_items(state, previous_state, previous_top_addresses)
            if not feature_items:
                previous_state = state
                continue

            address_weights: Dict[int, float] = defaultdict(float)
            gate_values: Dict[int, float] = {}
            for address_id, feature_weight, feature_name in feature_items:
                proto_vec = self._complex_feature_vector(feature_name)
                score = _complex_alignment(context_vec, proto_vec)
                gate_signal = 3.2 * score.real + 1.4 * abs(score) + 0.6 * math.cos(math.atan2(score.imag, score.real + 1e-12))
                gate = _sigmoid(gate_signal)
                address_weights[address_id] += float(feature_weight * gate)
                gate_values[address_id] = max(gate_values.get(address_id, 0.0), gate)

            trimmed = _top_items(dict(address_weights), 24)
            previous_top_addresses = list(trimmed.keys())[:10]
            activations.append(
                ChunkActivation(
                    address_weights=trimmed,
                    gate_values={addr: gate_values[addr] for addr in trimmed},
                    chunk_text=state.chunk_text,
                )
            )
            previous_state = state
        return activations

    def _state_context_vector(self, prefix_states: Sequence[ChunkState]) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.complex64)
        if not prefix_states:
            return vec
        recent_states = list(prefix_states[-8:])
        weights: Dict[str, float] = defaultdict(float)
        for idx, state in enumerate(recent_states):
            recency = 1.0 + 0.45 * idx / max(len(recent_states), 1)
            for sem in state.semantic_keys[:3]:
                weights[f"ctx_sem::{sem}"] += 1.2 * recency * self.idf.get(f"sem::{sem}", 1.0)
            for rel in state.relation_keys[:3]:
                weights[f"ctx_rel::{rel}"] += 1.4 * recency * self.idf.get(f"rel::{rel}", 1.0)
            for sem in state.semantic_keys[:2]:
                for rel in state.relation_keys[:2]:
                    feat = f"ctx_node::{sem}::{rel}"
                    weights[feat] += 1.8 * recency * self.idf.get(f"pair::{sem}::{rel}", 1.0)
            for flag in state.flags[:2]:
                feat = f"ctx_flag::{flag}"
                weights[feat] += 1.5 * recency * self.idf.get(f"flag::{flag}", 1.0)
            for marker in state.causal_markers[:2]:
                feat = f"ctx_cause::{marker}"
                weights[feat] += 1.7 * recency * self.idf.get(f"cause::{marker}", 1.0)
            for marker in state.turn_markers[:2]:
                feat = f"ctx_turn::{marker}"
                weights[feat] += 1.7 * recency * self.idf.get(f"turn::{marker}", 1.0)
            for role_feat in state.event_roles[:4]:
                feat = f"ctx_role::{role_feat}"
                weights[feat] += 1.9 * recency
            for schema in state.causal_schemas[:3]:
                feat = f"ctx_schema::{schema}"
                weights[feat] += 2.0 * recency
            for act in state.narrative_acts[:3]:
                feat = f"ctx_act::{act}"
                weights[feat] += 1.8 * recency
        for left, right in zip(recent_states[:-1], recent_states[1:]):
            for prev_rel in left.relation_keys[:2]:
                for rel in right.relation_keys[:2]:
                    feat = f"ctx_edge_rel::{prev_rel}=>{rel}"
                    weights[feat] += 2.0 * self.idf.get(f"rel::{rel}", 1.0)
            for prev_sem in left.semantic_keys[:2]:
                for sem in right.semantic_keys[:2]:
                    feat = f"ctx_edge_sem::{prev_sem}=>{sem}"
                    weights[feat] += 1.7 * self.idf.get(f"sem::{sem}", 1.0)
            for prev_sem in left.semantic_keys[:1]:
                for prev_rel in left.relation_keys[:1]:
                    for sem in right.semantic_keys[:1]:
                        for rel in right.relation_keys[:1]:
                            feat = f"ctx_edge_node::{prev_sem}/{prev_rel}=>{sem}/{rel}"
                            weights[feat] += 2.4
            for left_role in left.event_roles[:2]:
                for right_role in right.event_roles[:2]:
                    feat = f"ctx_edge_role::{left_role}=>{right_role}"
                    weights[feat] += 2.2
            for left_schema in left.causal_schemas[:2]:
                for right_schema in right.causal_schemas[:2]:
                    feat = f"ctx_edge_schema::{left_schema}=>{right_schema}"
                    weights[feat] += 2.4
            for left_act in left.narrative_acts[:2]:
                for right_act in right.narrative_acts[:2]:
                    feat = f"ctx_edge_act::{left_act}=>{right_act}"
                    weights[feat] += 2.3
        for feat, weight in weights.items():
            vec += np.float32(weight) * self._complex_feature_vector(feat)
        return _normalize_complex(vec)

    def _address_feature_items(
        self,
        state: ChunkState,
        previous_state: ChunkState | None,
        previous_top_addresses: Sequence[int],
    ) -> List[Tuple[int, float, str]]:
        items: List[Tuple[int, float, str]] = []
        current_nodes = [
            (sem, rel)
            for sem in state.semantic_keys[:3]
            for rel in state.relation_keys[:3]
        ]
        for sem, rel in current_nodes:
            feat = f"node::{sem}::{rel}"
            weight = 2.8 * state.weight * self.idf.get(f"pair::{sem}::{rel}", 1.0)
            items.append((stable_hash(feat, self.address_count), float(weight), feat))

        for role_feat in state.event_roles:
            feat = f"role::{role_feat}"
            items.append((stable_hash(feat, self.address_count), float(2.2 * state.weight), feat))

        for schema in state.causal_schemas[:4]:
            feat = f"schema::{schema}"
            items.append((stable_hash(feat, self.address_count), float(2.4 * state.weight), feat))

        for act in state.narrative_acts[:4]:
            feat = f"act::{act}"
            items.append((stable_hash(feat, self.address_count), float(2.3 * state.weight), feat))

        for marker in state.causal_markers[:3]:
            feat = f"cause::{marker}"
            items.append(
                (
                    stable_hash(feat, self.address_count),
                    float(2.1 * state.weight * self.idf.get(feat, 1.0)),
                    feat,
                )
            )

        for marker in state.turn_markers[:3]:
            feat = f"turn::{marker}"
            items.append(
                (
                    stable_hash(feat, self.address_count),
                    float(2.1 * state.weight * self.idf.get(feat, 1.0)),
                    feat,
                )
            )

        sem_bundle = "+".join(sorted(state.semantic_keys[:2]))
        rel_bundle = "+".join(sorted(state.relation_keys[:2]))
        bundle_feat = f"state::{sem_bundle}::{rel_bundle}"
        items.append((stable_hash(bundle_feat, self.address_count), float(2.2 * state.weight), bundle_feat))

        for flag in state.flags[:3]:
            for rel in state.relation_keys[:2]:
                feat = f"flagrel::{flag}::{rel}"
                items.append((stable_hash(feat, self.address_count), float(1.9 * state.weight), feat))

        if previous_state is not None:
            for prev_rel in previous_state.relation_keys[:2]:
                for rel in state.relation_keys[:2]:
                    feat = f"edge_rel::{prev_rel}=>{rel}"
                    weight = 2.4 * self.idf.get(f"rel::{rel}", 1.0)
                    if prev_rel != rel:
                        weight *= 1.15
                    items.append((stable_hash(feat, self.address_count), float(weight), feat))
            for prev_sem in previous_state.semantic_keys[:2]:
                for sem in state.semantic_keys[:2]:
                    feat = f"edge_sem::{prev_sem}=>{sem}"
                    weight = 2.1 * self.idf.get(f"sem::{sem}", 1.0)
                    if prev_sem == sem:
                        weight *= 1.2
                    items.append((stable_hash(feat, self.address_count), float(weight), feat))
            for prev_sem in previous_state.semantic_keys[:1]:
                for prev_rel in previous_state.relation_keys[:1]:
                    for sem in state.semantic_keys[:1]:
                        for rel in state.relation_keys[:1]:
                            feat = f"edge_node::{prev_sem}/{prev_rel}=>{sem}/{rel}"
                            weight = 3.0
                            if prev_sem == sem and prev_rel != rel:
                                weight *= 1.25
                            items.append((stable_hash(feat, self.address_count), float(weight), feat))
            shared_sem = sorted(set(previous_state.semantic_keys[:3]).intersection(state.semantic_keys[:3]))
            for sem in shared_sem[:2]:
                for prev_rel in previous_state.relation_keys[:1]:
                    for rel in state.relation_keys[:1]:
                        feat = f"reframe::{sem}::{prev_rel}=>{rel}"
                        weight = 2.6 if prev_rel != rel else 2.1
                        items.append((stable_hash(feat, self.address_count), float(weight), feat))
            for prev_marker in previous_state.causal_markers[:2]:
                for marker in state.causal_markers[:2]:
                    feat = f"edge_cause::{prev_marker}=>{marker}"
                    items.append((stable_hash(feat, self.address_count), 2.3, feat))
            for prev_turn in previous_state.turn_markers[:2]:
                for turn in state.turn_markers[:2]:
                    feat = f"edge_turn::{prev_turn}=>{turn}"
                    weight = 2.4 if prev_turn != turn else 2.0
                    items.append((stable_hash(feat, self.address_count), weight, feat))
            for prev_role in previous_state.event_roles[:2]:
                for role in state.event_roles[:2]:
                    feat = f"edge_role::{prev_role}=>{role}"
                    items.append((stable_hash(feat, self.address_count), 2.5, feat))
            for prev_schema in previous_state.causal_schemas[:2]:
                for schema in state.causal_schemas[:2]:
                    feat = f"edge_schema::{prev_schema}=>{schema}"
                    items.append((stable_hash(feat, self.address_count), 2.7, feat))
            for prev_act in previous_state.narrative_acts[:2]:
                for act in state.narrative_acts[:2]:
                    feat = f"edge_act::{prev_act}=>{act}"
                    weight = 2.6 if prev_act != act else 2.2
                    items.append((stable_hash(feat, self.address_count), weight, feat))

        for prev_address in previous_top_addresses[:8]:
            feat = f"carry_state::{prev_address}::{rel_bundle}"
            items.append((stable_hash(feat, self.address_count), float(0.7 * state.weight), feat))
        return items

    def _complex_feature_vector(self, feature_name: str) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.complex64)
        base_angle = (2.0 * math.pi * stable_hash(f"phase::{feature_name}", 8192)) / 8192.0
        for lane in range(8):
            idx = stable_hash(f"{feature_name}::lane::{lane}", self.context_dim)
            lane_angle = base_angle + (2.0 * math.pi * lane / 8.0)
            sign = -1.0 if stable_hash(f"{feature_name}::sign::{lane}", 2) == 0 else 1.0
            vec[idx] += np.complex64(sign * complex(math.cos(lane_angle), math.sin(lane_angle)))
        return _normalize_complex(vec)


class LLMStateAddressEngram(StateAddressEngram):
    def __init__(
        self,
        address_count: int = 4096,
        context_dim: int = 256,
        ngram_orders: Sequence[int] = (2, 3),
        hash_heads: int = 4,
        chunk_sentences: int = 2,
        projector_shape: Tuple[int, int, int] = (12, 12, 12),
        llm_model: str = "claude-haiku-4-5-20251001",
        llm_api_key: str = "",
        llm_backend: str = "anthropic",
        llm_cache_dir: Path | None = None,
    ) -> None:
        super().__init__(
            address_count=address_count,
            context_dim=context_dim,
            ngram_orders=ngram_orders,
            hash_heads=hash_heads,
            chunk_sentences=chunk_sentences,
            projector_shape=projector_shape,
        )
        self.extractor = LLMStateExtractor(
            model=llm_model,
            api_key=llm_api_key,
            backend=llm_backend,
            cache_dir=llm_cache_dir,
            chunk_sentences=chunk_sentences,
        )

    def extract_states(self, text: str) -> List[ChunkState]:
        llm_states = self.extractor.extract_chunk_states(text)
        mapped_states = llm_chunk_states_to_state_fields(llm_states)
        return [ChunkState(**item) for item in mapped_states]


class LLMConstraintObjectAddressEngram(LLMStateAddressEngram):
    def extract_states(self, text: str) -> List[ChunkState]:
        constraint_steps = self.extractor.extract_constraint_steps(text)
        mapped_states = llm_constraint_steps_to_state_fields(constraint_steps)
        return [ChunkState(**item) for item in mapped_states]


class LLMConstraintEdgeAddressEngram(LLMConstraintObjectAddressEngram):
    @staticmethod
    def _prefixed_values(values: Sequence[str], prefix: str, fallback_prefix: str = "") -> List[str]:
        found: List[str] = []
        prefixes = [prefix]
        if fallback_prefix:
            prefixes.append(fallback_prefix)
        for value in values:
            lowered = value.lower()
            for current_prefix in prefixes:
                if lowered.startswith(current_prefix):
                    found.append(lowered.split("::", 1)[1])
                    break
        return list(dict.fromkeys(item for item in found if item))[:4]

    def _current_state_token(self, state: ChunkState) -> str:
        state_tokens = self._prefixed_values(state.event_roles, "state::")
        if state_tokens:
            return state_tokens[0]
        for sem in state.semantic_keys:
            lowered = sem.lower()
            if lowered.startswith("state::"):
                return lowered.split("::", 1)[1]
        return (state.semantic_keys[:1] or ["context"])[0].lower()

    def _preferred_next_tokens(self, state: ChunkState) -> List[str]:
        relation_tokens = self._prefixed_values(state.relation_keys, "allow::")
        role_tokens = self._prefixed_values(state.event_roles, "prefer::")
        return list(dict.fromkeys(relation_tokens + role_tokens))[:4] or ["persistence"]

    def _forbidden_next_tokens(self, state: ChunkState) -> List[str]:
        relation_tokens = self._prefixed_values(state.relation_keys, "block::")
        role_tokens = self._prefixed_values(state.event_roles, "forbid::")
        return list(dict.fromkeys(relation_tokens + role_tokens))[:4]

    def _required_constraint_tokens(self, state: ChunkState) -> List[str]:
        relation_tokens = self._prefixed_values(state.relation_keys, "require::")
        schema_tokens = self._prefixed_values(state.causal_schemas, "require::")
        return list(dict.fromkeys(relation_tokens + schema_tokens))[:4]

    def _trigger_tokens(self, state: ChunkState) -> List[str]:
        turn_tokens = self._prefixed_values(state.turn_markers, "trigger::")
        role_tokens = self._prefixed_values(state.event_roles, "trigger::")
        act_tokens = self._prefixed_values(state.narrative_acts, "trigger::")
        return list(dict.fromkeys(turn_tokens + role_tokens + act_tokens))[:4] or ["persistence"]

    def _constraint_edge_records_for_state(
        self,
        step_idx: int,
        state: ChunkState,
        previous_state: ChunkState | None,
    ) -> List[ConstraintEdgeRecord]:
        current = self._current_state_token(state)
        preferred = self._preferred_next_tokens(state)
        forbidden = self._forbidden_next_tokens(state)
        required = self._required_constraint_tokens(state)
        triggers = self._trigger_tokens(state)
        records: List[ConstraintEdgeRecord] = []
        for nxt in preferred[:4]:
            records.append(
                ConstraintEdgeRecord(
                    step_idx=step_idx,
                    source_state=current,
                    edge_type="allow",
                    target=nxt,
                    weight=3.2 * state.weight * (1.0 + 0.08 * len(required)),
                )
            )
        for nxt in forbidden[:3]:
            records.append(
                ConstraintEdgeRecord(
                    step_idx=step_idx,
                    source_state=current,
                    edge_type="block",
                    target=nxt,
                    weight=2.9 * state.weight * (1.0 + 0.06 * len(forbidden)),
                )
            )
        for req in required[:3]:
            records.append(
                ConstraintEdgeRecord(
                    step_idx=step_idx,
                    source_state=current,
                    edge_type="require",
                    target=req,
                    weight=2.7 * state.weight,
                )
            )
        for trigger in triggers[:3]:
            records.append(
                ConstraintEdgeRecord(
                    step_idx=step_idx,
                    source_state=current,
                    edge_type="trigger",
                    target=current,
                    trigger=trigger,
                    weight=2.5 * state.weight,
                )
            )
        if previous_state is not None:
            previous_current = self._current_state_token(previous_state)
            prev_preferred = self._preferred_next_tokens(previous_state)
            records.append(
                ConstraintEdgeRecord(
                    step_idx=step_idx,
                    source_state=previous_current,
                    edge_type="step",
                    target=current,
                    weight=2.8,
                )
            )
            for prev_next in prev_preferred[:2]:
                records.append(
                    ConstraintEdgeRecord(
                        step_idx=step_idx,
                        source_state=previous_current,
                        edge_type="flow",
                        target=current,
                        trigger=prev_next,
                        status="realized" if prev_next == current else "outside",
                        weight=3.0 if prev_next == current else 2.4,
                    )
                )
            for prev_block in self._forbidden_next_tokens(previous_state)[:2]:
                if prev_block == current:
                    records.append(
                        ConstraintEdgeRecord(
                            step_idx=step_idx,
                            source_state=previous_current,
                            edge_type="conflict",
                            target=current,
                            trigger=prev_block,
                            status="blocked",
                            weight=2.7,
                        )
                    )
        return records

    def _all_constraint_edge_records(self, states: Sequence[ChunkState]) -> List[ConstraintEdgeRecord]:
        records: List[ConstraintEdgeRecord] = []
        previous_state: ChunkState | None = None
        for step_idx, state in enumerate(states):
            records.extend(self._constraint_edge_records_for_state(step_idx, state, previous_state))
            previous_state = state
        return records

    def _state_context_vector(self, prefix_states: Sequence[ChunkState]) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.complex64)
        if not prefix_states:
            return vec
        recent_states = list(prefix_states[-8:])
        weights: Dict[str, float] = defaultdict(float)
        previous_state: ChunkState | None = None
        for idx, state in enumerate(recent_states):
            recency = 1.0 + 0.45 * idx / max(len(recent_states), 1)
            records = self._constraint_edge_records_for_state(idx, state, previous_state)
            weights[f"ctx_constraint_state::{self._current_state_token(state)}"] += 2.0 * recency
            for record in records:
                if record.edge_type == "allow":
                    feat = f"ctx_allow_edge::{record.source_state}=>{record.target}"
                elif record.edge_type == "block":
                    feat = f"ctx_block_edge::{record.source_state}-/->{record.target}"
                elif record.edge_type == "require":
                    feat = f"ctx_require_edge::{record.source_state}=>{record.target}"
                elif record.edge_type == "trigger":
                    feat = f"ctx_trigger_edge::{record.trigger}::{record.source_state}"
                elif record.edge_type == "step":
                    feat = f"ctx_constraint_step::{record.source_state}=>{record.target}"
                elif record.edge_type == "flow":
                    feat = f"ctx_constraint_flow::{record.source_state}=>{record.trigger}=>{record.target}"
                else:
                    feat = f"ctx_constraint_conflict::{record.source_state}-/->{record.trigger}::{record.target}"
                weights[feat] += recency * record.weight
            previous_state = state
        for feat, weight in weights.items():
            vec += np.float32(weight) * self._complex_feature_vector(feat)
        return _normalize_complex(vec)

    def _address_feature_items(
        self,
        state: ChunkState,
        previous_state: ChunkState | None,
        previous_top_addresses: Sequence[int],
    ) -> List[Tuple[int, float, str]]:
        items: List[Tuple[int, float, str]] = []
        current = self._current_state_token(state)
        records = self._constraint_edge_records_for_state(len(previous_top_addresses), state, previous_state)

        state_feat = f"constraint_state::{current}"
        items.append((stable_hash(state_feat, self.address_count), float(2.8 * state.weight), state_feat))
        preferred = self._preferred_next_tokens(state)
        for record in records:
            if record.edge_type == "allow":
                feat = f"allow_edge::{record.source_state}=>{record.target}"
            elif record.edge_type == "block":
                feat = f"block_edge::{record.source_state}-/->{record.target}"
            elif record.edge_type == "require":
                feat = f"require_edge::{record.source_state}=>{record.target}"
            elif record.edge_type == "trigger":
                feat = f"trigger_edge::{record.trigger}::{record.source_state}"
            elif record.edge_type == "step":
                feat = f"constraint_step::{record.source_state}=>{record.target}"
            elif record.edge_type == "flow":
                feat = f"constraint_flow::{record.source_state}=>{record.trigger}=>{record.target}"
            else:
                feat = f"constraint_conflict::{record.source_state}-/->{record.trigger}::{record.target}"
            items.append((stable_hash(feat, self.address_count), float(record.weight), feat))
            if record.edge_type == "trigger":
                for nxt in preferred[:2]:
                    combo_feat = f"trigger_allow::{record.trigger}::{current}=>{nxt}"
                    items.append((stable_hash(combo_feat, self.address_count), float(2.7 * state.weight), combo_feat))

        for prev_address in previous_top_addresses[:8]:
            feat = f"carry_constraint::{prev_address}::{current}"
            items.append((stable_hash(feat, self.address_count), float(0.75 * state.weight), feat))
        return items


class LLMConstraintEdgeVolumeEngram(LLMConstraintEdgeAddressEngram):
    def __init__(
        self,
        address_count: int = 4096,
        context_dim: int = 256,
        ngram_orders: Sequence[int] = (2, 3),
        hash_heads: int = 4,
        chunk_sentences: int = 2,
        projector_shape: Tuple[int, int, int] = (12, 12, 12),
        llm_model: str = "claude-haiku-4-5-20251001",
        llm_api_key: str = "",
        llm_backend: str = "anthropic",
        llm_cache_dir: Path | None = None,
    ) -> None:
        super().__init__(
            address_count=address_count,
            context_dim=context_dim,
            ngram_orders=ngram_orders,
            hash_heads=hash_heads,
            chunk_sentences=chunk_sentences,
            projector_shape=projector_shape,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        )
        self.volume_projector = ConstraintEdgeVolumeProjector(shape=(24, 24, 24))
        self.codec = K3MCodec(
            base_radii=(1, 1, 1),
            tube_half_length=2,
            tube_radius=1,
            max_tubes=12,
            max_junctions=8,
            residual_budget=128,
            response_steps=2,
        )
        self.last_encode_stats: Dict[str, float] = {}

    def encode_states(self, states: Sequence[ChunkState]) -> List[ChunkActivation]:
        if not states:
            self.last_encode_stats = {}
            return []
        edge_records = self._all_constraint_edge_records(states)
        volume, patterns, projector_stats = self.volume_projector.project(edge_records, total_steps=len(states))
        archive = self.codec.encode(volume)
        recon = self.codec.decode(archive)
        patterns_by_step: Dict[int, List[object]] = defaultdict(list)
        for pattern in patterns:
            patterns_by_step[pattern.step_idx].append(pattern)
        activations: List[ChunkActivation] = []
        for step_idx, state in enumerate(states):
            address_weights: Dict[int, float] = defaultdict(float)
            gate_values: Dict[int, float] = {}
            for pattern in patterns_by_step.get(step_idx, []):
                mask = pattern.support_mask
                if not np.any(mask):
                    continue
                response = float(np.mean(np.abs(recon[mask])))
                gate = _sigmoid(2.4 * response)
                address_id = stable_hash(f"native::{pattern.address_token}", self.address_count)
                address_weights[address_id] += float(pattern.weight * gate)
                gate_values[address_id] = max(gate_values.get(address_id, 0.0), gate)
            trimmed = _top_items(dict(address_weights), 24)
            activations.append(
                ChunkActivation(
                    address_weights=trimmed,
                    gate_values={addr: gate_values[addr] for addr in trimmed},
                    chunk_text=state.chunk_text,
                )
            )
        raw_bytes = len(" ".join(state.chunk_text for state in states).encode("utf-8"))
        archive_bytes = archive.compressed_size_bytes()
        self.last_encode_stats = {
            "archive_bytes": float(archive_bytes),
            "compression_ratio": float(raw_bytes / max(archive_bytes, 1)),
            "tube_count": float(len(archive.tubes)),
            "junction_count": float(len(archive.junctions)),
            "support_ratio": float(archive.metadata.get("support_ratio", 0.0)),
            "direction_coverage_ratio": float(archive.metadata.get("direction_coverage_ratio", 0.0)),
            "pattern_count": float(projector_stats.get("pattern_count", 0.0)),
            "junction_pattern_count": float(projector_stats.get("junction_pattern_count", 0.0)),
        }
        return activations


class LLMRotatingPhaseStateAddressEngram(LLMStateAddressEngram):
    def __init__(
        self,
        address_count: int = 4096,
        context_dim: int = 256,
        ngram_orders: Sequence[int] = (2, 3),
        hash_heads: int = 4,
        chunk_sentences: int = 2,
        projector_shape: Tuple[int, int, int] = (12, 12, 12),
        llm_model: str = "claude-haiku-4-5-20251001",
        llm_api_key: str = "",
        llm_backend: str = "anthropic",
        llm_cache_dir: Path | None = None,
        phase_decay: float = 0.22,
    ) -> None:
        super().__init__(
            address_count=address_count,
            context_dim=context_dim,
            ngram_orders=ngram_orders,
            hash_heads=hash_heads,
            chunk_sentences=chunk_sentences,
            projector_shape=projector_shape,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        )
        self.phase_decay = float(phase_decay)

    def encode_states(self, states: Sequence[ChunkState]) -> List[ChunkActivation]:
        if not states:
            return []
        phase_history = self._phase_history(states)
        activations: List[ChunkActivation] = []
        prefix_states: List[ChunkState] = []
        previous_state: ChunkState | None = None
        previous_top_addresses: List[int] = []
        for idx, state in enumerate(states):
            prefix_states.append(state)
            context_vec = self._state_context_vector(prefix_states, phase_history[: idx + 1])
            feature_items = self._address_feature_items(state, previous_state, previous_top_addresses)
            if not feature_items:
                previous_state = state
                continue

            state_phase = phase_history[idx]
            phase_scale = float(math.exp(-self.phase_decay * idx))
            address_weights: Dict[int, float] = defaultdict(float)
            gate_values: Dict[int, float] = {}
            for address_id, feature_weight, feature_name in feature_items:
                proto_vec = self._rotate_complex_vector(
                    self._complex_feature_vector(feature_name),
                    state_phase,
                    phase_scale,
                )
                score = _complex_alignment(context_vec, proto_vec)
                gate_signal = (
                    2.9 * score.real
                    + 1.7 * abs(score)
                    + 0.9 * math.cos(math.atan2(score.imag, score.real + 1e-12))
                )
                gate = _sigmoid(gate_signal)
                address_weights[address_id] += float(feature_weight * gate)
                gate_values[address_id] = max(gate_values.get(address_id, 0.0), gate)

            trimmed = _top_items(dict(address_weights), 24)
            previous_top_addresses = list(trimmed.keys())[:10]
            activations.append(
                ChunkActivation(
                    address_weights=trimmed,
                    gate_values={addr: gate_values[addr] for addr in trimmed},
                    chunk_text=state.chunk_text,
                )
            )
            previous_state = state
        return activations

    def _state_context_vector(self, prefix_states: Sequence[ChunkState], phase_history: Sequence[float]) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.complex64)
        if not prefix_states:
            return vec
        recent_states = list(prefix_states[-8:])
        recent_phases = list(phase_history[-8:])
        weights: Dict[str, complex] = defaultdict(complex)
        total_recent = max(len(recent_states), 1)
        for idx, (state, phase) in enumerate(zip(recent_states, recent_phases)):
            recency = 1.0 + 0.45 * idx / total_recent
            decay = math.exp(-self.phase_decay * (total_recent - 1 - idx))
            multiplier = decay * complex(math.cos(phase), math.sin(phase))
            for sem in state.semantic_keys[:3]:
                weights[f"ctx_sem::{sem}"] += multiplier * (1.2 * recency * self.idf.get(f"sem::{sem}", 1.0))
            for rel in state.relation_keys[:3]:
                weights[f"ctx_rel::{rel}"] += multiplier * (1.4 * recency * self.idf.get(f"rel::{rel}", 1.0))
            for tone in state.emotions[:3]:
                weights[f"ctx_tone::{tone}"] += multiplier * (1.9 * recency)
            for turn in state.turn_markers[:3]:
                weights[f"ctx_turn::{turn}"] += multiplier * (1.9 * recency)
            for sem in state.semantic_keys[:2]:
                for rel in state.relation_keys[:2]:
                    feat = f"ctx_node::{sem}::{rel}"
                    weights[feat] += multiplier * (2.0 * recency * self.idf.get(f"pair::{sem}::{rel}", 1.0))
        for idx, (left, right) in enumerate(zip(recent_states[:-1], recent_states[1:])):
            phase = 0.5 * (recent_phases[idx] + recent_phases[idx + 1])
            decay = math.exp(-self.phase_decay * (len(recent_states) - 2 - idx))
            multiplier = decay * complex(math.cos(phase), math.sin(phase))
            for prev_rel in left.relation_keys[:2]:
                for rel in right.relation_keys[:2]:
                    weights[f"ctx_edge_rel::{prev_rel}=>{rel}"] += multiplier * (2.1 * self.idf.get(f"rel::{rel}", 1.0))
            for prev_sem in left.semantic_keys[:2]:
                for sem in right.semantic_keys[:2]:
                    weights[f"ctx_edge_sem::{prev_sem}=>{sem}"] += multiplier * (1.9 * self.idf.get(f"sem::{sem}", 1.0))
            for prev_turn in left.turn_markers[:2]:
                for turn in right.turn_markers[:2]:
                    weights[f"ctx_edge_turn::{prev_turn}=>{turn}"] += multiplier * 2.2
            for prev_tone in left.emotions[:2]:
                for tone in right.emotions[:2]:
                    weights[f"ctx_edge_tone::{prev_tone}=>{tone}"] += multiplier * 2.0
        for feat, weight in weights.items():
            vec += np.complex64(weight) * self._complex_feature_vector(feat)
        return _normalize_complex(vec)

    def _phase_history(self, states: Sequence[ChunkState]) -> List[float]:
        history: List[float] = []
        running_phase = 0.0
        previous_state: ChunkState | None = None
        for idx, state in enumerate(states):
            running_phase += self._phase_increment(state, previous_state, idx)
            history.append(running_phase)
            previous_state = state
        return history

    def _phase_increment(self, state: ChunkState, previous_state: ChunkState | None, idx: int) -> float:
        time_term = math.exp(-self.phase_decay * idx)
        topic_angle = self._topic_angle(state)
        turn_shift = self._turn_shift(state, previous_state)
        tone_shift = self._tone_shift(state, previous_state)
        change_shift = self._topic_change_shift(state, previous_state)
        return time_term * (0.55 * topic_angle + turn_shift + tone_shift + change_shift)

    def _topic_angle(self, state: ChunkState) -> float:
        topic_key = "|".join(sorted(state.semantic_keys[:3])) or "misc"
        return (2.0 * math.pi * stable_hash(f"topic-phase::{topic_key}", 8192)) / 8192.0

    def _turn_shift(self, state: ChunkState, previous_state: ChunkState | None) -> float:
        shift = 0.0
        turns = {item.lower() for item in state.turn_markers[:3] + state.narrative_acts[:3]}
        if "turn" in turns or "contrast" in turns or "reframe" in turns:
            shift += 0.75
        if "resolution" in turns or "commitment" in turns:
            shift += 0.35
        if previous_state is not None and set(previous_state.turn_markers[:2]) != set(state.turn_markers[:2]):
            shift += 0.45
        return shift

    def _tone_shift(self, state: ChunkState, previous_state: ChunkState | None) -> float:
        tone_weights = {
            "contrastive": 0.65,
            "assertive": 0.45,
            "uncertain": -0.35,
            "reflective": 0.20,
            "tense": 0.55,
            "positive": 0.25,
        }
        current_tones = [tone.lower() for tone in state.emotions[:4]]
        shift = sum(tone_weights.get(tone, 0.0) for tone in current_tones)
        if previous_state is not None:
            prev_tones = {tone.lower() for tone in previous_state.emotions[:4]}
            shift += 0.35 * len(set(current_tones).difference(prev_tones))
        return shift

    def _topic_change_shift(self, state: ChunkState, previous_state: ChunkState | None) -> float:
        if previous_state is None:
            return 0.0
        prev_topics = set(previous_state.semantic_keys[:3])
        curr_topics = set(state.semantic_keys[:3])
        overlap = len(prev_topics.intersection(curr_topics))
        novel = len(curr_topics.difference(prev_topics))
        return 0.40 * novel - 0.18 * overlap

    def _rotate_complex_vector(self, vec: np.ndarray, phase: float, scale: float) -> np.ndarray:
        multiplier = np.complex64(scale * complex(math.cos(phase), math.sin(phase)))
        return _normalize_complex(vec * multiplier)


class LLMEventPhaseStateAddressEngram(LLMRotatingPhaseStateAddressEngram):
    def __init__(
        self,
        address_count: int = 4096,
        context_dim: int = 256,
        ngram_orders: Sequence[int] = (2, 3),
        hash_heads: int = 4,
        chunk_sentences: int = 2,
        projector_shape: Tuple[int, int, int] = (12, 12, 12),
        llm_model: str = "claude-haiku-4-5-20251001",
        llm_api_key: str = "",
        llm_backend: str = "anthropic",
        llm_cache_dir: Path | None = None,
        phase_decay: float = 0.22,
        phase_bucket_count: int = 8,
    ) -> None:
        super().__init__(
            address_count=address_count,
            context_dim=context_dim,
            ngram_orders=ngram_orders,
            hash_heads=hash_heads,
            chunk_sentences=chunk_sentences,
            projector_shape=projector_shape,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
            phase_decay=phase_decay,
        )
        self.phase_bucket_count = int(phase_bucket_count)

    def encode_states(self, states: Sequence[ChunkState]) -> List[ChunkActivation]:
        if not states:
            return []
        phase_history = self._phase_history(states)
        phase_buckets = [self._phase_bucket(phase) for phase in phase_history]
        activations: List[ChunkActivation] = []
        prefix_states: List[ChunkState] = []
        previous_state: ChunkState | None = None
        previous_top_addresses: List[int] = []
        for idx, state in enumerate(states):
            prefix_states.append(state)
            state_phase = phase_history[idx]
            phase_bucket = phase_buckets[idx]
            context_vec = self._event_state_context_vector(
                prefix_states,
                phase_history[: idx + 1],
                phase_buckets[: idx + 1],
            )
            feature_items = self._event_address_feature_items(
                state,
                previous_state,
                previous_top_addresses,
                phase_bucket,
            )
            if not feature_items:
                previous_state = state
                continue

            phase_scale = float(math.exp(-self.phase_decay * idx))
            address_weights: Dict[int, float] = defaultdict(float)
            gate_values: Dict[int, float] = {}
            for address_id, feature_weight, feature_name in feature_items:
                proto_vec = self._rotate_complex_vector(
                    self._complex_feature_vector(feature_name),
                    state_phase,
                    phase_scale,
                )
                score = _complex_alignment(context_vec, proto_vec)
                gate_signal = (
                    3.1 * score.real
                    + 1.8 * abs(score)
                    + 1.1 * math.cos(math.atan2(score.imag, score.real + 1e-12))
                )
                gate = _sigmoid(gate_signal)
                address_weights[address_id] += float(feature_weight * gate)
                gate_values[address_id] = max(gate_values.get(address_id, 0.0), gate)

            trimmed = _top_items(dict(address_weights), 24)
            previous_top_addresses = list(trimmed.keys())[:10]
            activations.append(
                ChunkActivation(
                    address_weights=trimmed,
                    gate_values={addr: gate_values[addr] for addr in trimmed},
                    chunk_text=state.chunk_text,
                )
            )
            previous_state = state
        return activations

    def _phase_increment(self, state: ChunkState, previous_state: ChunkState | None, idx: int) -> float:
        time_term = math.exp(-self.phase_decay * idx)
        event_strength = self._event_trigger_strength(state, previous_state)
        if event_strength <= 1e-6:
            return 0.05 * time_term
        event_bucket = self._event_bucket(state, previous_state)
        bucket_angle = (2.0 * math.pi * (event_bucket + 0.5)) / max(self.phase_bucket_count, 1)
        return time_term * (0.12 + 0.55 * event_strength + 0.18 * bucket_angle)

    def _event_trigger_strength(self, state: ChunkState, previous_state: ChunkState | None) -> float:
        strength = 0.0
        turns = {item.lower() for item in state.turn_markers[:3] + state.narrative_acts[:3]}
        tones = {item.lower() for item in state.emotions[:4]}
        rels = {item.lower() for item in state.relation_keys[:4]}
        if turns.intersection({"turn", "contrast", "reframe", "redirect"}):
            strength += 1.2
        if turns.intersection({"resolution", "commitment", "inquiry"}):
            strength += 0.6
        if tones.intersection({"contrastive", "assertive", "uncertain", "tense"}):
            strength += 0.9
        if rels.intersection({"topic_shift", "reframe", "question_to_resolution", "cause_to_decision"}):
            strength += 1.0
        if previous_state is not None:
            if set(previous_state.semantic_keys[:3]) != set(state.semantic_keys[:3]):
                strength += 0.7
            if set(previous_state.turn_markers[:3]) != set(state.turn_markers[:3]):
                strength += 0.7
            if set(previous_state.emotions[:3]) != set(state.emotions[:3]):
                strength += 0.5
        return strength

    def _event_bucket(self, state: ChunkState, previous_state: ChunkState | None) -> int:
        prev_turns = ",".join(sorted(previous_state.turn_markers[:2])) if previous_state is not None else "init"
        prev_topics = ",".join(sorted(previous_state.semantic_keys[:2])) if previous_state is not None else "init"
        signature = "|".join(
            [
                ",".join(sorted(state.turn_markers[:3])),
                ",".join(sorted(state.emotions[:3])),
                ",".join(sorted(state.relation_keys[:3])),
                ",".join(sorted(state.semantic_keys[:3])),
                prev_turns,
                prev_topics,
            ]
        )
        return stable_hash(f"event-phase::{signature}", self.phase_bucket_count)

    def _phase_bucket(self, phase: float) -> int:
        wrapped = (phase % (2.0 * math.pi)) / (2.0 * math.pi)
        return min(self.phase_bucket_count - 1, int(math.floor(wrapped * self.phase_bucket_count)))

    def _event_state_context_vector(
        self,
        prefix_states: Sequence[ChunkState],
        phase_history: Sequence[float],
        phase_buckets: Sequence[int],
    ) -> np.ndarray:
        vec = super()._state_context_vector(prefix_states, phase_history)
        recent_states = list(prefix_states[-8:])
        recent_phases = list(phase_history[-8:])
        recent_buckets = list(phase_buckets[-8:])
        total_recent = max(len(recent_states), 1)
        for idx, (state, phase, bucket) in enumerate(zip(recent_states, recent_phases, recent_buckets)):
            decay = math.exp(-self.phase_decay * (total_recent - 1 - idx))
            multiplier = np.complex64(decay * complex(math.cos(phase), math.sin(phase)))
            vec += multiplier * 1.3 * self._complex_feature_vector(f"ctx_phasebucket::{bucket}")
            for turn in state.turn_markers[:2]:
                vec += multiplier * 1.7 * self._complex_feature_vector(f"ctx_phase_turn::{bucket}::{turn}")
            for tone in state.emotions[:2]:
                vec += multiplier * 1.6 * self._complex_feature_vector(f"ctx_phase_tone::{bucket}::{tone}")
            for rel in state.relation_keys[:2]:
                vec += multiplier * 1.5 * self._complex_feature_vector(f"ctx_phase_rel::{bucket}::{rel}")
        return _normalize_complex(vec)

    def _event_address_feature_items(
        self,
        state: ChunkState,
        previous_state: ChunkState | None,
        previous_top_addresses: Sequence[int],
        phase_bucket: int,
    ) -> List[Tuple[int, float, str]]:
        items = list(super()._address_feature_items(state, previous_state, previous_top_addresses))
        for sem in state.semantic_keys[:2]:
            for rel in state.relation_keys[:2]:
                feat = f"phase_node::{phase_bucket}::{sem}::{rel}"
                items.append((stable_hash(feat, self.address_count), float(2.7 * state.weight), feat))
        for turn in state.turn_markers[:3]:
            feat = f"phase_turn::{phase_bucket}::{turn}"
            items.append((stable_hash(feat, self.address_count), float(2.4 * state.weight), feat))
        for tone in state.emotions[:3]:
            feat = f"phase_tone::{phase_bucket}::{tone}"
            items.append((stable_hash(feat, self.address_count), float(2.2 * state.weight), feat))
        for act in state.narrative_acts[:2]:
            feat = f"phase_act::{phase_bucket}::{act}"
            items.append((stable_hash(feat, self.address_count), float(2.3 * state.weight), feat))
        if previous_state is not None:
            prev_bucket = self._event_bucket(previous_state, None)
            if prev_bucket != phase_bucket:
                feat = f"phase_jump::{prev_bucket}=>{phase_bucket}"
                items.append((stable_hash(feat, self.address_count), float(3.0 * state.weight), feat))
        return items


class LLMEventTransitionAddressEngram(LLMEventPhaseStateAddressEngram):
    def encode_states(self, states: Sequence[ChunkState]) -> List[ChunkActivation]:
        if not states:
            return []
        event_states = self._event_transition_states(states)
        activations: List[ChunkActivation] = []
        history: List[EventTransitionState] = []
        previous_event: EventTransitionState | None = None
        previous_top_addresses: List[int] = []
        for event_state in event_states:
            history.append(event_state)
            context_vec = self._event_transition_context_vector(history, previous_top_addresses)
            feature_items = self._event_transition_feature_items(event_state, previous_event, previous_top_addresses)
            address_weights: Dict[int, float] = defaultdict(float)
            gate_values: Dict[int, float] = {}
            for address_id, feature_weight, feature_name in feature_items:
                proto_vec = self._event_transition_proto_vector(feature_name, event_state.phase_bucket)
                score = _complex_alignment(context_vec, proto_vec)
                gate_signal = 3.4 * score.real + 2.0 * abs(score)
                gate = _sigmoid(gate_signal)
                address_weights[address_id] += float(feature_weight * gate)
                gate_values[address_id] = max(gate_values.get(address_id, 0.0), gate)
            trimmed = _top_items(dict(address_weights), 24)
            previous_top_addresses = list(trimmed.keys())[:10]
            activations.append(
                ChunkActivation(
                    address_weights=trimmed,
                    gate_values={addr: gate_values[addr] for addr in trimmed},
                    chunk_text=" ".join(event_state.event_tags[:3]),
                )
            )
            previous_event = event_state
        return activations

    def _event_transition_states(self, states: Sequence[ChunkState]) -> List[EventTransitionState]:
        event_states: List[EventTransitionState] = []
        previous_state: ChunkState | None = None
        for state in states:
            event_tags = self._event_transition_tags(state, previous_state)
            strength = self._event_trigger_strength(state, previous_state)
            phase_bucket = self._event_bucket(state, previous_state)
            tone_shift = self._tone_shift_label(state, previous_state)
            topic_shift = self._topic_shift_label(state, previous_state)
            causal_shift = self._causal_shift_label(state, previous_state)
            relation_signature = self._relation_signature(state)
            narrative_signature = self._narrative_signature(state)
            semantic_anchor = (sorted(state.semantic_keys[:1]) or ["misc"])[0]
            event_states.append(
                EventTransitionState(
                    phase_bucket=phase_bucket,
                    strength=strength,
                    event_tags=event_tags,
                    tone_shift=tone_shift,
                    topic_shift=topic_shift,
                    causal_shift=causal_shift,
                    relation_signature=relation_signature,
                    narrative_signature=narrative_signature,
                    semantic_anchor=semantic_anchor,
                )
            )
            previous_state = state
        return event_states

    def _event_transition_tags(self, state: ChunkState, previous_state: ChunkState | None) -> List[str]:
        tags: List[str] = []
        turns = {item.lower() for item in state.turn_markers[:3] + state.narrative_acts[:3]}
        tones = {item.lower() for item in state.emotions[:4]}
        rels = {item.lower() for item in state.relation_keys[:4]}
        if turns.intersection({"turn", "contrast", "reframe", "redirect"}):
            tags.append("turn_jump")
        if turns.intersection({"resolution", "commitment"}):
            tags.append("resolution_jump")
        if turns.intersection({"inquiry"}):
            tags.append("question_jump")
        if tones.intersection({"assertive", "contrastive", "uncertain", "tense"}):
            tags.append("tone_jump")
        if rels.intersection({"topic_shift", "reframe", "cause_to_decision", "question_to_resolution"}):
            tags.append("relation_jump")
        if previous_state is not None and set(previous_state.semantic_keys[:3]) != set(state.semantic_keys[:3]):
            tags.append("topic_jump")
        return tags or ["persistence"]

    def _tone_shift_label(self, state: ChunkState, previous_state: ChunkState | None) -> str:
        current = sorted(tone.lower() for tone in state.emotions[:3]) or ["flat"]
        previous = (
            sorted(tone.lower() for tone in previous_state.emotions[:3]) or ["flat"]
            if previous_state is not None
            else ["init"]
        )
        if current == previous:
            return current[0]
        return f"{previous[0]}=>{current[0]}"

    def _topic_shift_label(self, state: ChunkState, previous_state: ChunkState | None) -> str:
        if previous_state is None:
            return "init"
        prev_topics = set(previous_state.semantic_keys[:3])
        curr_topics = set(state.semantic_keys[:3])
        overlap = len(prev_topics.intersection(curr_topics))
        novel = len(curr_topics.difference(prev_topics))
        if novel >= 2 and overlap == 0:
            return "hard_shift"
        if novel >= 1 and overlap >= 1:
            return "soft_shift"
        if overlap >= 2:
            return "persist"
        return "mixed"

    def _causal_shift_label(self, state: ChunkState, previous_state: ChunkState | None) -> str:
        current = sorted(item.lower() for item in (state.causal_markers[:2] + state.causal_schemas[:2])) or ["background"]
        previous = (
            sorted(item.lower() for item in (previous_state.causal_markers[:2] + previous_state.causal_schemas[:2])) or ["background"]
            if previous_state is not None
            else ["init"]
        )
        if current == previous:
            return current[0]
        return f"{previous[0]}=>{current[0]}"

    def _relation_signature(self, state: ChunkState) -> str:
        return "+".join(sorted(item.lower() for item in state.relation_keys[:2])) or "context"

    def _narrative_signature(self, state: ChunkState) -> str:
        return "+".join(sorted(item.lower() for item in state.narrative_acts[:2])) or "setup"

    def _event_transition_context_vector(
        self,
        history: Sequence[EventTransitionState],
        previous_top_addresses: Sequence[int],
    ) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.complex64)
        recent = list(history[-8:])
        total_recent = max(len(recent), 1)
        for idx, event_state in enumerate(recent):
            decay = math.exp(-0.18 * (total_recent - 1 - idx))
            phase = (2.0 * math.pi * (event_state.phase_bucket + 0.5)) / self.phase_bucket_count
            multiplier = np.complex64(decay * complex(math.cos(phase), math.sin(phase)))
            vec += multiplier * (1.8 + event_state.strength) * self._complex_feature_vector(
                f"ctx_event_bucket::{event_state.phase_bucket}"
            )
            for tag in event_state.event_tags[:3]:
                vec += multiplier * 2.3 * self._complex_feature_vector(f"ctx_event_tag::{event_state.phase_bucket}::{tag}")
            vec += multiplier * 1.8 * self._complex_feature_vector(
                f"ctx_tone_shift::{event_state.phase_bucket}::{event_state.tone_shift}"
            )
            vec += multiplier * 1.8 * self._complex_feature_vector(
                f"ctx_topic_shift::{event_state.phase_bucket}::{event_state.topic_shift}"
            )
            vec += multiplier * 1.8 * self._complex_feature_vector(
                f"ctx_causal_shift::{event_state.phase_bucket}::{event_state.causal_shift}"
            )
            vec += multiplier * 1.3 * self._complex_feature_vector(
                f"ctx_sem_anchor::{event_state.semantic_anchor}"
            )
        for left, right in zip(recent[:-1], recent[1:]):
            phase = (2.0 * math.pi * ((left.phase_bucket + right.phase_bucket) / 2.0 + 0.5)) / self.phase_bucket_count
            multiplier = np.complex64(complex(math.cos(phase), math.sin(phase)))
            vec += multiplier * 2.8 * self._complex_feature_vector(
                f"ctx_event_edge::{left.event_tags[0]}=>{right.event_tags[0]}"
            )
            vec += multiplier * 2.4 * self._complex_feature_vector(
                f"ctx_bucket_edge::{left.phase_bucket}=>{right.phase_bucket}"
            )
        for prev_address in previous_top_addresses[:8]:
            vec += np.complex64(0.9) * self._complex_feature_vector(f"ctx_carry::{prev_address}")
        return _normalize_complex(vec)

    def _event_transition_feature_items(
        self,
        event_state: EventTransitionState,
        previous_event: EventTransitionState | None,
        previous_top_addresses: Sequence[int],
    ) -> List[Tuple[int, float, str]]:
        items: List[Tuple[int, float, str]] = []
        base_weight = 2.0 + event_state.strength
        items.append(self._event_item(f"event_bucket::{event_state.phase_bucket}", base_weight))
        for tag in event_state.event_tags[:4]:
            items.append(self._event_item(f"event_tag::{event_state.phase_bucket}::{tag}", base_weight + 0.8))
        items.append(self._event_item(f"tone_shift::{event_state.phase_bucket}::{event_state.tone_shift}", base_weight + 0.4))
        items.append(self._event_item(f"topic_shift::{event_state.phase_bucket}::{event_state.topic_shift}", base_weight + 0.6))
        items.append(self._event_item(f"causal_shift::{event_state.phase_bucket}::{event_state.causal_shift}", base_weight + 0.5))
        items.append(self._event_item(f"relation_sig::{event_state.phase_bucket}::{event_state.relation_signature}", base_weight))
        items.append(self._event_item(f"narrative_sig::{event_state.phase_bucket}::{event_state.narrative_signature}", base_weight))
        items.append(self._event_item(f"sem_anchor::{event_state.semantic_anchor}", 0.9))
        if previous_event is not None:
            items.append(
                self._event_item(
                    f"event_edge::{previous_event.event_tags[0]}=>{event_state.event_tags[0]}",
                    3.0 + 0.6 * event_state.strength,
                )
            )
            items.append(
                self._event_item(
                    f"bucket_edge::{previous_event.phase_bucket}=>{event_state.phase_bucket}",
                    2.8 + 0.4 * event_state.strength,
                )
            )
            items.append(
                self._event_item(
                    f"tone_edge::{previous_event.tone_shift}=>{event_state.tone_shift}",
                    2.4,
                )
            )
            items.append(
                self._event_item(
                    f"topic_edge::{previous_event.topic_shift}=>{event_state.topic_shift}",
                    2.5,
                )
            )
        for prev_address in previous_top_addresses[:6]:
            items.append(self._event_item(f"event_carry::{prev_address}::{event_state.phase_bucket}", 1.2))
        return items

    def _event_item(self, feature_name: str, weight: float) -> Tuple[int, float, str]:
        return (stable_hash(feature_name, self.address_count), float(weight), feature_name)

    def _event_transition_proto_vector(self, feature_name: str, phase_bucket: int) -> np.ndarray:
        phase = (2.0 * math.pi * (phase_bucket + 0.5)) / self.phase_bucket_count
        base_vec = self._complex_feature_vector(feature_name)
        return self._rotate_complex_vector(base_vec, phase, 1.0)


class LLMFiniteEventTransitionAddressEngram(LLMEventTransitionAddressEngram):
    EVENT_ALPHABET = (
        "persistence",
        "turn",
        "reframe",
        "topic_shift",
        "question",
        "resolution",
        "cause_to_decision",
        "tone_flip",
    )

    def _event_transition_states(self, states: Sequence[ChunkState]) -> List[EventTransitionState]:
        event_states: List[EventTransitionState] = []
        previous_state: ChunkState | None = None
        previous_event_label = "init"
        for state in states:
            event_label = self._finite_event_label(state, previous_state)
            phase_bucket = self.EVENT_ALPHABET.index(event_label) % self.phase_bucket_count
            strength = self._finite_event_strength(event_label, state, previous_state)
            relation_signature = self._finite_relation_signature(state)
            narrative_signature = event_label
            event_states.append(
                EventTransitionState(
                    phase_bucket=phase_bucket,
                    strength=strength,
                    event_tags=[event_label, f"{previous_event_label}=>{event_label}"],
                    tone_shift=self._tone_shift_label(state, previous_state),
                    topic_shift=self._topic_shift_label(state, previous_state),
                    causal_shift=self._causal_shift_label(state, previous_state),
                    relation_signature=relation_signature,
                    narrative_signature=narrative_signature,
                    semantic_anchor=(sorted(state.semantic_keys[:1]) or ["misc"])[0],
                )
            )
            previous_state = state
            previous_event_label = event_label
        return event_states

    def _finite_event_label(self, state: ChunkState, previous_state: ChunkState | None) -> str:
        turns = {item.lower() for item in state.turn_markers[:3] + state.narrative_acts[:3]}
        tones = {item.lower() for item in state.emotions[:4]}
        rels = {item.lower() for item in state.relation_keys[:4]}
        causals = {item.lower() for item in state.causal_markers[:3] + state.causal_schemas[:3]}
        if rels.intersection({"question_to_resolution"}) or turns.intersection({"resolution", "commitment"}):
            return "resolution"
        if rels.intersection({"cause_to_decision"}) or ("decision" in " ".join(state.flags).lower() and causals.intersection({"cause", "effect"})):
            return "cause_to_decision"
        if rels.intersection({"reframe"}) or turns.intersection({"reframe", "redirect", "contrast"}):
            return "reframe"
        if rels.intersection({"topic_shift"}) or self._topic_shift_label(state, previous_state) in {"hard_shift", "soft_shift"}:
            return "topic_shift"
        if turns.intersection({"inquiry"}) or rels.intersection({"question"}):
            return "question"
        if self._tone_shift_label(state, previous_state) not in {"init", "flat", "flat=>flat"} and tones.intersection(
            {"assertive", "contrastive", "uncertain", "tense"}
        ):
            return "tone_flip"
        if turns.intersection({"turn"}):
            return "turn"
        return "persistence"

    def _finite_event_strength(self, event_label: str, state: ChunkState, previous_state: ChunkState | None) -> float:
        base = {
            "persistence": 0.8,
            "turn": 1.8,
            "reframe": 2.4,
            "topic_shift": 2.2,
            "question": 1.7,
            "resolution": 2.0,
            "cause_to_decision": 2.3,
            "tone_flip": 1.6,
        }.get(event_label, 1.0)
        novelty = 0.0
        if previous_state is not None:
            novelty += 0.3 * len(set(state.semantic_keys[:3]).difference(previous_state.semantic_keys[:3]))
            novelty += 0.2 * len(set(state.relation_keys[:3]).difference(previous_state.relation_keys[:3]))
        return base + novelty

    def _finite_relation_signature(self, state: ChunkState) -> str:
        rels = {item.lower() for item in state.relation_keys[:4]}
        if rels.intersection({"cause_to_decision", "decision_to_effect", "decision_to_implementation"}):
            return "causal_transition"
        if rels.intersection({"reframe", "topic_shift"}):
            return "topic_transition"
        if rels.intersection({"question", "question_to_resolution"}):
            return "inquiry_transition"
        return "context_transition"

    def _event_transition_context_vector(
        self,
        history: Sequence[EventTransitionState],
        previous_top_addresses: Sequence[int],
    ) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.complex64)
        recent = list(history[-8:])
        total_recent = max(len(recent), 1)
        for idx, event_state in enumerate(recent):
            decay = math.exp(-0.22 * (total_recent - 1 - idx))
            phase = (2.0 * math.pi * (event_state.phase_bucket + 0.5)) / self.phase_bucket_count
            multiplier = np.complex64(decay * complex(math.cos(phase), math.sin(phase)))
            event_label = event_state.event_tags[0]
            vec += multiplier * (2.4 + event_state.strength) * self._complex_feature_vector(f"ctx_finite_event::{event_label}")
            vec += multiplier * 2.2 * self._complex_feature_vector(
                f"ctx_finite_relation::{event_label}::{event_state.relation_signature}"
            )
            vec += multiplier * 1.6 * self._complex_feature_vector(
                f"ctx_finite_topic::{event_label}::{event_state.topic_shift}"
            )
        for left, right in zip(recent[:-1], recent[1:]):
            left_label = left.event_tags[0]
            right_label = right.event_tags[0]
            vec += np.complex64(2.8) * self._complex_feature_vector(f"ctx_finite_edge::{left_label}=>{right_label}")
        for prev_address in previous_top_addresses[:8]:
            vec += np.complex64(0.6) * self._complex_feature_vector(f"ctx_finite_carry::{prev_address}")
        return _normalize_complex(vec)

    def _event_transition_feature_items(
        self,
        event_state: EventTransitionState,
        previous_event: EventTransitionState | None,
        previous_top_addresses: Sequence[int],
    ) -> List[Tuple[int, float, str]]:
        event_label = event_state.event_tags[0]
        items: List[Tuple[int, float, str]] = []
        items.append(self._event_item(f"finite_event::{event_label}", 3.0 + event_state.strength))
        items.append(self._event_item(f"finite_relation::{event_label}::{event_state.relation_signature}", 2.6))
        items.append(self._event_item(f"finite_topic::{event_label}::{event_state.topic_shift}", 2.2))
        items.append(self._event_item(f"finite_causal::{event_label}::{event_state.causal_shift}", 2.1))
        if previous_event is not None:
            prev_label = previous_event.event_tags[0]
            items.append(self._event_item(f"finite_edge::{prev_label}=>{event_label}", 3.6 + 0.4 * event_state.strength))
            items.append(
                self._event_item(
                    f"finite_pair::{prev_label}/{previous_event.relation_signature}=>{event_label}/{event_state.relation_signature}",
                    3.0,
                )
            )
        for prev_address in previous_top_addresses[:6]:
            items.append(self._event_item(f"finite_carry::{prev_address}::{event_label}", 1.0))
        return items


class CanonicalStateAddressEngram(StateAddressEngram):
    def fit(self, texts: Sequence[str]) -> "CanonicalStateAddressEngram":
        return self

    def _relation_family(self, label: str) -> str:
        text = label.lower()
        if any(key in text for key in ("decision", "choose", "prefer", "switch", "replace", "commit")):
            return "decision"
        if any(key in text for key in ("transition", "pivot", "change", "redirect", "turn")):
            return "transition"
        if any(key in text for key in ("origin", "genesis", "start", "create", "founded", "born")):
            return "origin"
        if any(key in text for key in ("question", "ask", "uncertain", "inquiry")):
            return "question"
        if any(key in text for key in ("principle", "core", "fundamental", "essential")):
            return "principle"
        if any(key in text for key in ("cause", "effect", "condition", "because", "since", "thus")):
            return "causal"
        if any(key in text for key in ("build", "implement", "update", "deploy", "migrate", "fix")):
            return "action"
        return "context"

    def _canonical_role(self, role: str) -> str:
        prefix, _, suffix = role.partition("::")
        if prefix in {"actor", "target", "modifier"}:
            return prefix
        if prefix == "action":
            return f"action_{self._relation_family(suffix)}"
        if prefix == "motive":
            return f"motive_{suffix or 'generic'}"
        if prefix == "turn":
            return f"turn_{suffix or 'generic'}"
        if prefix == "outcome":
            return f"outcome_{suffix or 'generic'}"
        return prefix or "role"

    def _canonical_schema(self, schema: str) -> str:
        return schema.split("::", 1)[-1]

    def _canonical_act(self, act: str) -> str:
        return act.split("::", 1)[-1]

    def _semantic_state_token(self, state: ChunkState, previous_state: ChunkState | None = None) -> str:
        if previous_state is None:
            return "semantic_init"
        overlap = set(previous_state.semantic_keys[:3]).intersection(state.semantic_keys[:3])
        if overlap:
            return "semantic_persist"
        return "semantic_shift"

    def _canonical_state_tokens(self, state: ChunkState, previous_state: ChunkState | None = None) -> List[str]:
        tokens: List[str] = []
        tokens.extend(f"relfam::{self._relation_family(rel)}" for rel in state.relation_keys[:3])
        tokens.extend(f"rolecanon::{self._canonical_role(role)}" for role in state.event_roles[:4])
        tokens.extend(f"schemacanon::{self._canonical_schema(schema)}" for schema in state.causal_schemas[:3])
        tokens.extend(f"actcanon::{self._canonical_act(act)}" for act in state.narrative_acts[:3])
        tokens.extend(f"flagcanon::{flag.lower()}" for flag in state.flags[:3])
        tokens.append(f"semstate::{self._semantic_state_token(state, previous_state)}")
        return list(dict.fromkeys(tokens))

    def _state_context_vector(self, prefix_states: Sequence[ChunkState]) -> np.ndarray:
        vec = np.zeros(self.context_dim, dtype=np.complex64)
        if not prefix_states:
            return vec
        recent_states = list(prefix_states[-8:])
        weights: Dict[str, float] = defaultdict(float)
        for idx, state in enumerate(recent_states):
            prev_state = recent_states[idx - 1] if idx > 0 else None
            recency = 1.0 + 0.45 * idx / max(len(recent_states), 1)
            for token in self._canonical_state_tokens(state, prev_state):
                weights[f"ctx::{token}"] += recency
        for left, right in zip(recent_states[:-1], recent_states[1:]):
            left_fams = [self._relation_family(rel) for rel in left.relation_keys[:2]] or ["context"]
            right_fams = [self._relation_family(rel) for rel in right.relation_keys[:2]] or ["context"]
            for src in left_fams:
                for dst in right_fams:
                    weights[f"ctx_edge_rel::{src}=>{dst}"] += 2.0
            left_roles = [self._canonical_role(role) for role in left.event_roles[:2]]
            right_roles = [self._canonical_role(role) for role in right.event_roles[:2]]
            for src in left_roles:
                for dst in right_roles:
                    weights[f"ctx_edge_role::{src}=>{dst}"] += 2.1
            left_schemas = [self._canonical_schema(schema) for schema in left.causal_schemas[:2]]
            right_schemas = [self._canonical_schema(schema) for schema in right.causal_schemas[:2]]
            for src in left_schemas:
                for dst in right_schemas:
                    weights[f"ctx_edge_schema::{src}=>{dst}"] += 2.2
            left_acts = [self._canonical_act(act) for act in left.narrative_acts[:2]]
            right_acts = [self._canonical_act(act) for act in right.narrative_acts[:2]]
            for src in left_acts:
                for dst in right_acts:
                    weights[f"ctx_edge_act::{src}=>{dst}"] += 2.0
            weights[f"ctx_edge_sem::{self._semantic_state_token(right, left)}"] += 1.8
        for feat, weight in weights.items():
            vec += np.float32(weight) * self._complex_feature_vector(feat)
        return _normalize_complex(vec)

    def _address_feature_items(
        self,
        state: ChunkState,
        previous_state: ChunkState | None,
        previous_top_addresses: Sequence[int],
    ) -> List[Tuple[int, float, str]]:
        items: List[Tuple[int, float, str]] = []
        relation_fams = [self._relation_family(rel) for rel in state.relation_keys[:3]] or ["context"]
        role_tokens = [self._canonical_role(role) for role in state.event_roles[:4]]
        schema_tokens = [self._canonical_schema(schema) for schema in state.causal_schemas[:3]]
        act_tokens = [self._canonical_act(act) for act in state.narrative_acts[:3]]
        sem_token = self._semantic_state_token(state, previous_state)

        for fam in relation_fams:
            feat = f"node_rel::{fam}"
            items.append((stable_hash(feat, self.address_count), 2.4 * state.weight, feat))
        for role in role_tokens:
            feat = f"node_role::{role}"
            items.append((stable_hash(feat, self.address_count), 2.1 * state.weight, feat))
        for schema in schema_tokens:
            feat = f"node_schema::{schema}"
            items.append((stable_hash(feat, self.address_count), 2.3 * state.weight, feat))
        for act in act_tokens:
            feat = f"node_act::{act}"
            items.append((stable_hash(feat, self.address_count), 2.2 * state.weight, feat))
        feat = f"node_sem::{sem_token}"
        items.append((stable_hash(feat, self.address_count), 1.8 * state.weight, feat))

        if previous_state is not None:
            prev_fams = [self._relation_family(rel) for rel in previous_state.relation_keys[:2]] or ["context"]
            for src in prev_fams:
                for dst in relation_fams[:2]:
                    feat = f"edge_rel::{src}=>{dst}"
                    items.append((stable_hash(feat, self.address_count), 2.6, feat))
            prev_roles = [self._canonical_role(role) for role in previous_state.event_roles[:2]]
            for src in prev_roles:
                for dst in role_tokens[:2]:
                    feat = f"edge_role::{src}=>{dst}"
                    items.append((stable_hash(feat, self.address_count), 2.5, feat))
            prev_schemas = [self._canonical_schema(schema) for schema in previous_state.causal_schemas[:2]]
            for src in prev_schemas:
                for dst in schema_tokens[:2]:
                    feat = f"edge_schema::{src}=>{dst}"
                    items.append((stable_hash(feat, self.address_count), 2.7, feat))
            prev_acts = [self._canonical_act(act) for act in previous_state.narrative_acts[:2]]
            for src in prev_acts:
                for dst in act_tokens[:2]:
                    feat = f"edge_act::{src}=>{dst}"
                    items.append((stable_hash(feat, self.address_count), 2.5, feat))
            feat = f"edge_sem::{self._semantic_state_token(state, previous_state)}"
            items.append((stable_hash(feat, self.address_count), 2.1, feat))

        for prev_address in previous_top_addresses[:8]:
            bundle = relation_fams[0] if relation_fams else "context"
            feat = f"carry_canonical::{prev_address}::{bundle}"
            items.append((stable_hash(feat, self.address_count), 0.6 * state.weight, feat))
        return items


class SemiCanonicalStateAddressEngram(CanonicalStateAddressEngram):
    def _relation_subtype(self, label: str) -> str:
        fam = self._relation_family(label)
        text = label.lower()
        if fam == "decision":
            if any(key in text for key in ("prefer", "choose")):
                return "preference"
            if any(key in text for key in ("switch", "replace", "migrate")):
                return "switch"
            return "commit"
        if fam == "transition":
            if any(key in text for key in ("pivot", "turn")):
                return "pivot"
            if any(key in text for key in ("contrast", "however", "but")):
                return "contrast"
            return "shift"
        if fam == "origin":
            return "setup"
        if fam == "question":
            return "inquiry"
        if fam == "principle":
            return "principle"
        if fam == "causal":
            return "causal"
        if fam == "action":
            return "implementation"
        return "context"

    def _semantic_band(self, semantic_key: str) -> str:
        key = semantic_key.lower()
        if key in {"misc", "context"}:
            return "generic"
        if len(key) <= 3:
            return "entity"
        if any(ch.isdigit() for ch in key):
            return "artifact"
        if "_" in key or "-" in key:
            return "compound"
        return key[:3]

    def _canonical_role(self, role: str) -> str:
        prefix, _, suffix = role.partition("::")
        if prefix == "actor":
            return f"actor_{self._semantic_band(suffix)}"
        if prefix == "target":
            return f"target_{self._semantic_band(suffix)}"
        if prefix == "modifier":
            return f"modifier_{self._relation_subtype(suffix)}"
        if prefix == "action":
            fam = self._relation_family(suffix)
            subtype = self._relation_subtype(suffix)
            return f"action_{fam}_{subtype}"
        if prefix == "motive":
            return f"motive_{suffix or 'generic'}"
        if prefix == "turn":
            return f"turn_{suffix or 'generic'}"
        if prefix == "outcome":
            return f"outcome_{suffix or 'generic'}"
        return prefix or "role"

    def _canonical_state_tokens(self, state: ChunkState, previous_state: ChunkState | None = None) -> List[str]:
        tokens: List[str] = []
        relation_fams = [self._relation_family(rel) for rel in state.relation_keys[:3]]
        relation_subtypes = [self._relation_subtype(rel) for rel in state.relation_keys[:3]]
        tokens.extend(f"relfam::{item}" for item in relation_fams)
        tokens.extend(f"relsub::{fam}::{sub}" for fam, sub in zip(relation_fams, relation_subtypes))
        tokens.extend(f"rolecanon::{self._canonical_role(role)}" for role in state.event_roles[:4])
        tokens.extend(f"schemacanon::{self._canonical_schema(schema)}" for schema in state.causal_schemas[:3])
        tokens.extend(f"actcanon::{self._canonical_act(act)}" for act in state.narrative_acts[:3])
        tokens.extend(f"flagcanon::{flag.lower()}" for flag in state.flags[:3])
        tokens.append(f"semstate::{self._semantic_state_token(state, previous_state)}")
        if state.semantic_keys:
            tokens.append(f"semband::{self._semantic_band(state.semantic_keys[0])}")
        return list(dict.fromkeys(tokens))

    def _address_feature_items(
        self,
        state: ChunkState,
        previous_state: ChunkState | None,
        previous_top_addresses: Sequence[int],
    ) -> List[Tuple[int, float, str]]:
        items: List[Tuple[int, float, str]] = []
        relation_fams = [self._relation_family(rel) for rel in state.relation_keys[:3]] or ["context"]
        relation_subtypes = [self._relation_subtype(rel) for rel in state.relation_keys[:3]] or ["context"]
        role_tokens = [self._canonical_role(role) for role in state.event_roles[:4]]
        schema_tokens = [self._canonical_schema(schema) for schema in state.causal_schemas[:3]]
        act_tokens = [self._canonical_act(act) for act in state.narrative_acts[:3]]
        sem_token = self._semantic_state_token(state, previous_state)
        sem_band = self._semantic_band(state.semantic_keys[0]) if state.semantic_keys else "generic"

        for fam, sub in zip(relation_fams[:3], relation_subtypes[:3]):
            feat = f"node_rel::{fam}::{sub}"
            items.append((stable_hash(feat, self.address_count), 2.3 * state.weight, feat))
        for role in role_tokens:
            feat = f"node_role::{role}"
            items.append((stable_hash(feat, self.address_count), 2.0 * state.weight, feat))
        for schema in schema_tokens:
            feat = f"node_schema::{schema}"
            items.append((stable_hash(feat, self.address_count), 2.2 * state.weight, feat))
        for act in act_tokens:
            feat = f"node_act::{act}"
            items.append((stable_hash(feat, self.address_count), 2.1 * state.weight, feat))
        feat = f"node_sem::{sem_token}::{sem_band}"
        items.append((stable_hash(feat, self.address_count), 1.9 * state.weight, feat))

        if previous_state is not None:
            prev_fams = [self._relation_family(rel) for rel in previous_state.relation_keys[:2]] or ["context"]
            prev_subs = [self._relation_subtype(rel) for rel in previous_state.relation_keys[:2]] or ["context"]
            for src_fam, src_sub in zip(prev_fams, prev_subs):
                for dst_fam, dst_sub in zip(relation_fams[:2], relation_subtypes[:2]):
                    feat = f"edge_rel::{src_fam}/{src_sub}=>{dst_fam}/{dst_sub}"
                    items.append((stable_hash(feat, self.address_count), 2.5, feat))
            prev_roles = [self._canonical_role(role) for role in previous_state.event_roles[:2]]
            for src in prev_roles:
                for dst in role_tokens[:2]:
                    feat = f"edge_role::{src}=>{dst}"
                    items.append((stable_hash(feat, self.address_count), 2.4, feat))
            prev_schemas = [self._canonical_schema(schema) for schema in previous_state.causal_schemas[:2]]
            for src in prev_schemas:
                for dst in schema_tokens[:2]:
                    feat = f"edge_schema::{src}=>{dst}"
                    items.append((stable_hash(feat, self.address_count), 2.6, feat))
            prev_acts = [self._canonical_act(act) for act in previous_state.narrative_acts[:2]]
            for src in prev_acts:
                for dst in act_tokens[:2]:
                    feat = f"edge_act::{src}=>{dst}"
                    items.append((stable_hash(feat, self.address_count), 2.4, feat))
            feat = f"edge_sem::{self._semantic_state_token(state, previous_state)}::{sem_band}"
            items.append((stable_hash(feat, self.address_count), 2.0, feat))

        for prev_address in previous_top_addresses[:8]:
            feat = f"carry_semicanon::{prev_address}::{relation_fams[0]}::{sem_band}"
            items.append((stable_hash(feat, self.address_count), 0.65 * state.weight, feat))
        return items


def shuffled_states(states: Sequence[ChunkState], seed: int) -> List[ChunkState]:
    shuffled = list(states)
    if len(shuffled) <= 1:
        return shuffled
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def relation_shuffled_states(states: Sequence[ChunkState], seed: int) -> List[ChunkState]:
    if len(states) <= 1:
        return list(states)
    rng = random.Random(seed)
    relation_pool = [list(state.relation_keys) for state in states]
    flag_pool = [list(state.flags) for state in states]
    emotion_pool = [list(state.emotions) for state in states]
    causal_pool = [list(state.causal_markers) for state in states]
    turn_pool = [list(state.turn_markers) for state in states]
    role_pool = [list(state.event_roles) for state in states]
    schema_pool = [list(state.causal_schemas) for state in states]
    act_pool = [list(state.narrative_acts) for state in states]
    rng.shuffle(relation_pool)
    rng.shuffle(flag_pool)
    rng.shuffle(emotion_pool)
    rng.shuffle(causal_pool)
    rng.shuffle(turn_pool)
    rng.shuffle(role_pool)
    rng.shuffle(schema_pool)
    rng.shuffle(act_pool)
    remapped: List[ChunkState] = []
    for idx, state in enumerate(states):
        new_relations = relation_pool[idx] or ["context"]
        new_flags = flag_pool[idx]
        new_emotions = emotion_pool[idx]
        new_causal = causal_pool[idx]
        new_turns = turn_pool[idx]
        new_roles = role_pool[idx]
        new_schemas = schema_pool[idx]
        new_acts = act_pool[idx]
        remapped.append(
            ChunkState(
                chunk_text=state.chunk_text,
                tokens=list(state.tokens),
                semantic_keys=list(state.semantic_keys),
                relation_keys=list(dict.fromkeys(new_relations))[:4] or ["context"],
                flags=list(dict.fromkeys(new_flags))[:4],
                emotions=list(dict.fromkeys(new_emotions))[:4],
                causal_markers=list(dict.fromkeys(new_causal))[:4],
                turn_markers=list(dict.fromkeys(new_turns))[:4],
                event_roles=list(dict.fromkeys(new_roles))[:8],
                causal_schemas=list(dict.fromkeys(new_schemas))[:6],
                narrative_acts=list(dict.fromkeys(new_acts))[:6],
                weight=state.weight,
            )
        )
    return remapped


def compute_macro_flow_metrics(states: Sequence[ChunkState]) -> Dict[str, float]:
    if len(states) <= 1:
        return {
            "semantic_flow_entropy_norm": 0.0,
            "semantic_flow_js_to_uniform": 0.0,
            "semantic_flow_top8_mass": 0.0,
            "relation_flow_entropy_norm": 0.0,
            "relation_flow_js_to_uniform": 0.0,
            "relation_flow_top8_mass": 0.0,
            "relation_self_repeat_share": 0.0,
        }

    semantic_flow: Counter[str] = Counter()
    relation_flow: Counter[str] = Counter()
    relation_self_repeat = 0.0
    relation_total = 0.0
    for left, right in zip(states[:-1], states[1:]):
        for src in left.semantic_keys[:2]:
            for dst in right.semantic_keys[:2]:
                semantic_flow[f"{src}=>{dst}"] += 1.0
        for src in left.relation_keys[:2]:
            for dst in right.relation_keys[:2]:
                relation_flow[f"{src}=>{dst}"] += 1.0
                relation_total += 1.0
                if src == dst:
                    relation_self_repeat += 1.0

    semantic_prob = _normalize_counter(semantic_flow)
    relation_prob = _normalize_counter(relation_flow)

    if semantic_prob.size:
        semantic_uniform = np.full(semantic_prob.size, 1.0 / semantic_prob.size, dtype=np.float64)
        semantic_entropy_norm = float(_entropy(semantic_prob) / max(math.log(semantic_prob.size), 1e-12))
        semantic_js = float(_js_divergence(semantic_prob, semantic_uniform))
        semantic_top8 = float(_top_mass(semantic_prob, 8))
    else:
        semantic_entropy_norm = 0.0
        semantic_js = 0.0
        semantic_top8 = 0.0

    if relation_prob.size:
        relation_uniform = np.full(relation_prob.size, 1.0 / relation_prob.size, dtype=np.float64)
        relation_entropy_norm = float(_entropy(relation_prob) / max(math.log(relation_prob.size), 1e-12))
        relation_js = float(_js_divergence(relation_prob, relation_uniform))
        relation_top8 = float(_top_mass(relation_prob, 8))
    else:
        relation_entropy_norm = 0.0
        relation_js = 0.0
        relation_top8 = 0.0

    return {
        "semantic_flow_entropy_norm": semantic_entropy_norm,
        "semantic_flow_js_to_uniform": semantic_js,
        "semantic_flow_top8_mass": semantic_top8,
        "relation_flow_entropy_norm": relation_entropy_norm,
        "relation_flow_js_to_uniform": relation_js,
        "relation_flow_top8_mass": relation_top8,
        "relation_self_repeat_share": float(relation_self_repeat / max(relation_total, 1e-12)),
    }


def _measure_from_activations(activation: ChunkActivation) -> Dict[int, float]:
    return {addr: max(float(weight), 0.0) for addr, weight in activation.address_weights.items() if weight > 1e-8}


def compute_metrics(
    activations: Sequence[ChunkActivation],
    address_count: int,
    top_k_active: int,
) -> Dict[str, float]:
    if not activations:
        return {
            "chunk_count": 0.0,
            "active_addresses": 0.0,
            "address_entropy_norm": 0.0,
            "address_js_to_uniform": 0.0,
            "top8_address_mass": 0.0,
            "gate": 0.0,
            "gate_gini": 0.0,
            "transition_entropy_norm": 0.0,
            "transition_js_to_uniform": 0.0,
            "top8_transition_mass": 0.0,
            "self_loop_share": 0.0,
            "consecutive_state_alignment": 0.0,
        }

    address_mass = np.zeros(address_count, dtype=np.float64)
    gate_values: List[float] = []
    active_sizes: List[float] = []
    measures: List[Dict[int, float]] = []
    transitions: Counter[Tuple[int, int]] = Counter()
    state_vectors: List[np.ndarray] = []
    self_loop_mass = 0.0
    total_transition_mass = 0.0

    for activation in activations:
        measure = _top_items(_measure_from_activations(activation), top_k_active)
        measures.append(measure)
        active_sizes.append(float(len(measure)))
        state_vec = np.zeros(address_count, dtype=np.float32)
        for addr, weight in measure.items():
            address_mass[addr] += weight
            state_vec[addr] = np.float32(weight)
            gate_values.append(float(activation.gate_values.get(addr, 0.0)))
        state_vectors.append(_normalize(state_vec))

    for left, right in zip(measures[:-1], measures[1:]):
        for src, src_weight in left.items():
            for dst, dst_weight in right.items():
                edge_weight = math.sqrt(max(src_weight * dst_weight, 0.0))
                transitions[(src, dst)] += edge_weight
                total_transition_mass += edge_weight
                if src == dst:
                    self_loop_mass += edge_weight

    address_prob = address_mass / max(float(np.sum(address_mass)), 1e-12)
    uniform_address = np.full(address_count, 1.0 / max(address_count, 1), dtype=np.float64)

    transition_values = np.array(list(transitions.values()), dtype=np.float64)
    transition_total = float(np.sum(transition_values))
    if transition_total > 1e-12:
        transition_prob = transition_values / transition_total
        uniform_transition = np.full(transition_prob.size, 1.0 / transition_prob.size, dtype=np.float64)
        transition_entropy_norm = float(_entropy(transition_prob) / max(math.log(transition_prob.size), 1e-12))
        transition_js_to_uniform = float(_js_divergence(transition_prob, uniform_transition))
        top8_transition_mass = float(_top_mass(transition_prob, 8))
    else:
        transition_entropy_norm = 0.0
        transition_js_to_uniform = 0.0
        top8_transition_mass = 0.0

    alignments = [
        _cosine_similarity(left, right)
        for left, right in zip(state_vectors[:-1], state_vectors[1:])
    ]

    return {
        "chunk_count": float(len(activations)),
        "active_addresses": float(mean(active_sizes)) if active_sizes else 0.0,
        "address_entropy_norm": float(_entropy(address_prob) / max(math.log(address_count), 1e-12)),
        "address_js_to_uniform": float(_js_divergence(address_prob, uniform_address)),
        "top8_address_mass": float(_top_mass(address_prob, 8)),
        "gate": float(mean(gate_values)) if gate_values else 0.0,
        "gate_gini": float(_gini(gate_values)),
        "transition_entropy_norm": transition_entropy_norm,
        "transition_js_to_uniform": transition_js_to_uniform,
        "top8_transition_mass": top8_transition_mass,
        "self_loop_share": float(self_loop_mass / max(total_transition_mass, 1e-12)),
        "consecutive_state_alignment": float(mean(alignments)) if alignments else 0.0,
    }


def aggregate(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {f"avg_{key}": float(mean(item[key] for item in metrics)) for key in keys}


def build_observer(
    mode: str,
    texts: Sequence[str],
    address_count: int = 4096,
    context_dim: int = 256,
    hash_heads: int = 4,
    chunk_sentences: int = 2,
    llm_model: str = "claude-haiku-4-5-20251001",
    llm_api_key: str = "",
    llm_backend: str = "anthropic",
    llm_cache_dir: Path | None = None,
):
    common_kwargs = {
        "address_count": address_count,
        "context_dim": context_dim,
        "hash_heads": hash_heads,
        "chunk_sentences": chunk_sentences,
    }
    if mode == "semi-canonical-state-address":
        return SemiCanonicalStateAddressEngram(**common_kwargs).fit(texts)
    if mode == "canonical-state-address":
        return CanonicalStateAddressEngram(**common_kwargs).fit(texts)
    if mode == "state-address":
        return StateAddressEngram(**common_kwargs).fit(texts)
    if mode == "llm-state-address":
        return LLMStateAddressEngram(
            **common_kwargs,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        ).fit(texts)
    if mode == "llm-constraint-object-address":
        return LLMConstraintObjectAddressEngram(
            **common_kwargs,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        ).fit(texts)
    if mode == "llm-constraint-edge-address":
        return LLMConstraintEdgeAddressEngram(
            **common_kwargs,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        ).fit(texts)
    if mode == "llm-constraint-edge-volume":
        return LLMConstraintEdgeVolumeEngram(
            **common_kwargs,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        ).fit(texts)
    if mode == "llm-rotating-phase-state-address":
        return LLMRotatingPhaseStateAddressEngram(
            **common_kwargs,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        ).fit(texts)
    if mode == "llm-event-phase-state-address":
        return LLMEventPhaseStateAddressEngram(
            **common_kwargs,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        ).fit(texts)
    if mode == "llm-event-transition-address":
        return LLMEventTransitionAddressEngram(
            **common_kwargs,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        ).fit(texts)
    if mode == "llm-finite-event-transition-address":
        return LLMFiniteEventTransitionAddressEngram(
            **common_kwargs,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_backend=llm_backend,
            llm_cache_dir=llm_cache_dir,
        ).fit(texts)
    if mode == "relation-aware":
        return RelationAwareObservableEngram(**common_kwargs).fit(texts)
    if mode == "naive":
        return ObservableEngram(**common_kwargs).fit(texts)
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Directional nontriviality test in Engram address-transition space.")
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval_s_cleaned.json"))
    parser.add_argument("--limit-questions", type=int, default=100)
    parser.add_argument("--limit-sessions", type=int, default=256)
    parser.add_argument("--address-count", type=int, default=4096)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--hash-heads", type=int, default=4)
    parser.add_argument("--chunk-sentences", type=int, default=2)
    parser.add_argument("--top-k-active", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=[
            "naive",
            "relation-aware",
            "state-address",
            "canonical-state-address",
            "semi-canonical-state-address",
            "llm-state-address",
            "llm-constraint-object-address",
            "llm-constraint-edge-address",
            "llm-constraint-edge-volume",
            "llm-rotating-phase-state-address",
            "llm-event-phase-state-address",
            "llm-event-transition-address",
            "llm-finite-event-transition-address",
        ],
        default="state-address",
    )
    parser.add_argument("--llm-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-backend", choices=["anthropic", "cursor-agent", "auto"], default="anthropic")
    parser.add_argument("--llm-cache-dir", type=Path, default=Path(".cache/llm_state_extractor"))
    parser.add_argument("--json-out", type=Path, default=Path("engram_directional_nontriviality_results.json"))
    args = parser.parse_args()

    data = load_longmemeval(args.data, limit_questions=args.limit_questions)
    sessions = build_session_catalog(data)
    session_items = list(sessions.items())[: args.limit_sessions]
    texts = [record.text for _, record in session_items]

    observer = build_observer(
        mode=args.mode,
        texts=texts,
        address_count=args.address_count,
        context_dim=args.context_dim,
        hash_heads=args.hash_heads,
        chunk_sentences=args.chunk_sentences,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_backend=args.llm_backend,
        llm_cache_dir=args.llm_cache_dir,
    )

    original_metrics: List[Dict[str, float]] = []
    shuffled_metrics: List[Dict[str, float]] = []
    relation_shuffled_metrics: List[Dict[str, float]] = []
    original_macro_metrics: List[Dict[str, float]] = []
    shuffled_macro_metrics: List[Dict[str, float]] = []
    relation_shuffled_macro_metrics: List[Dict[str, float]] = []

    def _metric_with_observer_stats(activations: Sequence[ChunkActivation]) -> Dict[str, float]:
        metric = compute_metrics(activations, args.address_count, args.top_k_active)
        extra = getattr(observer, "last_encode_stats", {})
        if isinstance(extra, dict) and extra:
            metric.update({key: float(value) for key, value in extra.items()})
        return metric

    for idx, (_session_id, record) in enumerate(session_items):
        if args.mode in STATE_BASED_MODES:
            states = observer.extract_states(record.text)
            original_activations = observer.encode_states(states)
            original_metric = _metric_with_observer_stats(original_activations)
            shuffled_state_seq = shuffled_states(states, seed=idx)
            relation_shuffled_state_seq = relation_shuffled_states(states, seed=10_000 + idx)
            shuffled_activations = observer.encode_states(shuffled_state_seq)
            shuffled_metric = _metric_with_observer_stats(shuffled_activations)
            relation_only_activations = observer.encode_states(relation_shuffled_state_seq)
            relation_metric = _metric_with_observer_stats(relation_only_activations)
            original_macro_metrics.append(compute_macro_flow_metrics(states))
            shuffled_macro_metrics.append(compute_macro_flow_metrics(shuffled_state_seq))
            relation_shuffled_macro_metrics.append(compute_macro_flow_metrics(relation_shuffled_state_seq))
        else:
            original_activations = observer.encode_text(record.text)
            original_metric = _metric_with_observer_stats(original_activations)
            shuffled_text = shuffled_chunk_text(record.text, seed=idx)
            shuffled_activations = observer.encode_text(shuffled_text)
            shuffled_metric = _metric_with_observer_stats(shuffled_activations)
            relation_only_activations = shuffled_activations
            relation_metric = shuffled_metric

        original_metrics.append(original_metric)
        shuffled_metrics.append(shuffled_metric)
        relation_shuffled_metrics.append(relation_metric)

    original_summary = aggregate(original_metrics)
    shuffled_summary = aggregate(shuffled_metrics)
    relation_shuffled_summary = aggregate(relation_shuffled_metrics)
    original_macro_summary = aggregate(original_macro_metrics)
    shuffled_macro_summary = aggregate(shuffled_macro_metrics)
    relation_shuffled_macro_summary = aggregate(relation_shuffled_macro_metrics)
    deltas = {
        key.replace("avg_", "delta_"): float(original_summary[key] - shuffled_summary[key])
        for key in original_summary
        if key in shuffled_summary
    }
    relation_deltas = {
        key.replace("avg_", "delta_rel_"): float(original_summary[key] - relation_shuffled_summary[key])
        for key in original_summary
        if key in relation_shuffled_summary
    }
    macro_deltas = {
        key.replace("avg_", "delta_macro_"): float(original_macro_summary[key] - shuffled_macro_summary[key])
        for key in original_macro_summary
        if key in shuffled_macro_summary
    }
    relation_macro_deltas = {
        key.replace("avg_", "delta_macro_rel_"): float(original_macro_summary[key] - relation_shuffled_macro_summary[key])
        for key in original_macro_summary
        if key in relation_shuffled_macro_summary
    }

    verdict = {
        "nontrivial_directionality": bool(
            (
                deltas.get("delta_transition_js_to_uniform", 0.0) > 0.005
                and deltas.get("delta_top8_transition_mass", 0.0) > 0.005
                and relation_deltas.get("delta_rel_transition_js_to_uniform", 0.0) > 0.003
            )
            or (
                macro_deltas.get("delta_macro_relation_flow_js_to_uniform", 0.0) > 0.01
                and relation_macro_deltas.get("delta_macro_rel_relation_flow_js_to_uniform", 0.0) > 0.01
            )
        ),
        "notes": {
            "higher_transition_js_to_uniform_means_less_random_transition_flow": True,
            "higher_top8_transition_mass_means_more_concentrated_activation_paths": True,
            "higher_consecutive_state_alignment_means_more_coherent_address_evolution": True,
            "relation_shuffle_tests_whether_relation_labels_carry_extra_directional_signal": True,
            "macro_flow_metrics_check_coarse_semantic_relation_transition_structure": True,
        },
    }

    payload = {
        "config": {
            "questions": len(data),
            "sessions": len(session_items),
            "address_count": args.address_count,
            "context_dim": args.context_dim,
            "hash_heads": args.hash_heads,
            "chunk_sentences": args.chunk_sentences,
            "top_k_active": args.top_k_active,
            "mode": args.mode,
        },
        "original_summary": original_summary,
        "shuffled_summary": shuffled_summary,
        "relation_shuffled_summary": relation_shuffled_summary,
        "original_macro_summary": original_macro_summary,
        "shuffled_macro_summary": shuffled_macro_summary,
        "relation_shuffled_macro_summary": relation_shuffled_macro_summary,
        "deltas": deltas,
        "relation_deltas": relation_deltas,
        "macro_deltas": macro_deltas,
        "relation_macro_deltas": relation_macro_deltas,
        "verdict": verdict,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 96)
    print("Engram Directional Nontriviality")
    print("=" * 96)
    print("[original]")
    for key, value in original_summary.items():
        print(f"{key}: {value:.4f}")
    print("[shuffled_chunk_control]")
    for key, value in shuffled_summary.items():
        print(f"{key}: {value:.4f}")
    print("[relation_shuffled_control]")
    for key, value in relation_shuffled_summary.items():
        print(f"{key}: {value:.4f}")
    if original_macro_summary:
        print("[macro original]")
        for key, value in original_macro_summary.items():
            print(f"{key}: {value:.4f}")
        print("[macro shuffled_chunk_control]")
        for key, value in shuffled_macro_summary.items():
            print(f"{key}: {value:.4f}")
        print("[macro relation_shuffled_control]")
        for key, value in relation_shuffled_macro_summary.items():
            print(f"{key}: {value:.4f}")
    print("[delta original - shuffled]")
    for key, value in deltas.items():
        print(f"{key}: {value:.4f}")
    print("[delta original - relation_shuffled]")
    for key, value in relation_deltas.items():
        print(f"{key}: {value:.4f}")
    if macro_deltas:
        print("[macro delta original - shuffled]")
        for key, value in macro_deltas.items():
            print(f"{key}: {value:.4f}")
        print("[macro delta original - relation_shuffled]")
        for key, value in relation_macro_deltas.items():
            print(f"{key}: {value:.4f}")
    print(f"nontrivial_directionality: {verdict['nontrivial_directionality']}")
    print("=" * 96)
    print(f"[done] wrote {args.json_out}")


if __name__ == "__main__":
    main()
