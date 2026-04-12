from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from k3m_codec import K3MCodec, MemoryQuery


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
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

EMOTION_SIGNALS = {
    "decided": "determ",
    "prefer": "convict",
    "worried": "anx",
    "excited": "excite",
    "frustrated": "frust",
    "confused": "confuse",
    "love": "love",
    "hate": "rage",
    "hope": "hope",
    "fear": "fear",
    "trust": "trust",
    "happy": "joy",
    "sad": "grief",
    "surprised": "surprise",
    "grateful": "grat",
    "curious": "curious",
    "wonder": "wonder",
    "anxious": "anx",
    "relieved": "relief",
    "concern": "anx",
}

FLAG_SIGNALS = {
    "decided": "DECISION",
    "chose": "DECISION",
    "switched": "DECISION",
    "migrated": "DECISION",
    "replaced": "DECISION",
    "instead of": "DECISION",
    "because": "DECISION",
    "founded": "ORIGIN",
    "created": "ORIGIN",
    "started": "ORIGIN",
    "born": "ORIGIN",
    "launched": "ORIGIN",
    "core": "CORE",
    "fundamental": "CORE",
    "essential": "CORE",
    "principle": "CORE",
    "sensitive": "SENSITIVE",
    "private": "SENSITIVE",
    "confidential": "SENSITIVE",
    "turning point": "PIVOT",
    "pivot": "PIVOT",
    "changed everything": "PIVOT",
    "breakthrough": "PIVOT",
    "genesis": "GENESIS",
}

K_METRICS = (1, 3, 5, 10)


@dataclass
class SessionRecord:
    session_id: str
    date: str
    text: str


@dataclass
class AAAKDoc:
    session_id: str
    compressed_text: str
    vector: np.ndarray
    raw_bytes: int
    compressed_bytes: int


@dataclass
class K3MDoc:
    session_id: str
    archive: object
    vector: np.ndarray
    raw_bytes: int
    archive_bytes: int
    encode_seconds: float
    decode_seconds: float


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"[.!?\n]+", text)
    return [part.strip() for part in parts if len(part.strip()) > 3]


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class AAAKLiteDialect:
    def _detect_entities(self, text: str) -> List[str]:
        found: List[str] = []
        words = text.split()
        for i, word in enumerate(words):
            clean = re.sub(r"[^A-Za-z]", "", word)
            if (
                len(clean) >= 2
                and clean[0].isupper()
                and clean[1:].islower()
                and i > 0
                and clean.lower() not in STOP_WORDS
            ):
                code = clean[:3].upper()
                if code not in found:
                    found.append(code)
            if len(found) >= 3:
                break
        return found

    def _extract_topics(self, text: str, max_topics: int = 3) -> List[str]:
        freq: Counter[str] = Counter()
        for word in TOKEN_RE.findall(text):
            lowered = word.lower()
            if lowered in STOP_WORDS or len(lowered) < 3:
                continue
            weight = 1
            if word[0].isupper():
                weight += 2
            if "_" in word or "-" in word or any(ch.isupper() for ch in word[1:]):
                weight += 2
            freq[lowered] += weight
        return [word for word, _ in freq.most_common(max_topics)]

    def _extract_key_sentence(self, text: str) -> str:
        decision_words = {
            "decided",
            "because",
            "instead",
            "prefer",
            "switched",
            "chose",
            "realized",
            "important",
            "key",
            "critical",
            "discovered",
            "learned",
            "solution",
            "reason",
            "why",
            "breakthrough",
            "insight",
        }
        scored: List[Tuple[int, str]] = []
        for sentence in sentence_split(text):
            lowered = sentence.lower()
            score = sum(2 for word in decision_words if word in lowered)
            if len(sentence) < 80:
                score += 1
            if len(sentence) < 40:
                score += 1
            if len(sentence) > 150:
                score -= 2
            scored.append((score, sentence))
        if not scored:
            return ""
        best = sorted(scored, key=lambda item: item[0], reverse=True)[0][1]
        return best[:52] + "..." if len(best) > 55 else best

    def _detect_emotions(self, text: str) -> List[str]:
        lowered = text.lower()
        found: List[str] = []
        for keyword, code in EMOTION_SIGNALS.items():
            if keyword in lowered and code not in found:
                found.append(code)
            if len(found) >= 3:
                break
        return found

    def _detect_flags(self, text: str) -> List[str]:
        lowered = text.lower()
        found: List[str] = []
        for keyword, flag in FLAG_SIGNALS.items():
            if keyword in lowered and flag not in found:
                found.append(flag)
            if len(found) >= 3:
                break
        return found

    def compress(self, text: str, metadata: Optional[Dict[str, str]] = None) -> str:
        metadata = metadata or {}
        entities = self._detect_entities(text)
        topics = self._extract_topics(text)
        key_sentence = self._extract_key_sentence(text)
        emotions = self._detect_emotions(text)
        flags = self._detect_flags(text)

        entity_str = "+".join(entities[:3]) if entities else "???"
        topic_str = "_".join(topics[:3]) if topics else "misc"

        lines: List[str] = []
        if metadata.get("date"):
            lines.append(f"?|?|{metadata['date']}|?")

        parts = [f"0:{entity_str}", topic_str]
        if key_sentence:
            parts.append(f'"{key_sentence}"')
        if emotions:
            parts.append("+".join(emotions))
        if flags:
            parts.append("+".join(flags))
        lines.append("|".join(parts))
        return "\n".join(lines)


class HashedTextVectorizer:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.idf = np.ones(dim, dtype=np.float32)

    def features(self, text: str) -> List[str]:
        tokens = tokenize(text)
        features = list(tokens)
        features.extend(f"{a}__{b}" for a, b in zip(tokens[:-1], tokens[1:]))
        return features

    def fit(self, texts: Sequence[str]) -> None:
        doc_freq = np.zeros(self.dim, dtype=np.int32)
        total = len(texts)
        for text in texts:
            seen = set()
            for feature in self.features(text):
                seen.add(stable_hash(feature, self.dim))
            for idx in seen:
                doc_freq[idx] += 1
        self.idf = (np.log((1.0 + total) / (1.0 + doc_freq)) + 1.0).astype(np.float32)

    def transform(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature in self.features(text):
            idx = stable_hash(feature, self.dim)
            vec[idx] += 1.0
        if np.any(vec):
            vec = np.log1p(vec) * self.idf
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
        return vec.astype(np.float32)


class K3MTextProjector:
    def __init__(self, shape: Tuple[int, int, int]) -> None:
        self.shape = shape
        self.time_bins, self.u_bins, self.v_bins = shape
        self.idf = np.ones(self.u_bins * self.v_bins, dtype=np.float32)

    def feature_stream(self, text: str) -> List[str]:
        tokens = tokenize(text)
        features = list(tokens)
        features.extend(f"{a}__{b}" for a, b in zip(tokens[:-1], tokens[1:]))
        return features

    def fit(self, texts: Sequence[str]) -> None:
        doc_freq = np.zeros(self.u_bins * self.v_bins, dtype=np.int32)
        total = len(texts)
        for text in texts:
            seen = set()
            for feature in self.feature_stream(text):
                seen.add(stable_hash(feature, self.u_bins * self.v_bins))
            for idx in seen:
                doc_freq[idx] += 1
        self.idf = (np.log((1.0 + total) / (1.0 + doc_freq)) + 1.0).astype(np.float32)

    def text_to_volume(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        volume = np.zeros(self.shape, dtype=np.float32)
        if not tokens:
            return volume

        chunk_size = max(1, math.ceil(len(tokens) / self.time_bins))
        for token_idx, token in enumerate(tokens):
            t = min(token_idx // chunk_size, self.time_bins - 1)
            bucket = stable_hash(token, self.u_bins * self.v_bins)
            u = bucket // self.v_bins
            v = bucket % self.v_bins
            volume[t, u, v] += self.idf[bucket]

        for token_idx, (left, right) in enumerate(zip(tokens[:-1], tokens[1:])):
            t = min(token_idx // chunk_size, self.time_bins - 1)
            bucket = stable_hash(f"{left}__{right}", self.u_bins * self.v_bins)
            u = bucket // self.v_bins
            v = bucket % self.v_bins
            volume[t, u, v] += 0.5 * self.idf[bucket]

        if np.any(volume):
            volume /= np.linalg.norm(volume) + 1e-12
        return volume.astype(np.float32)

    def query_vector(self, text: str) -> np.ndarray:
        volume = self.text_to_volume(text)
        vec = volume.sum(axis=0).reshape(-1)
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)

    def doc_vector(self, recon_volume: np.ndarray) -> np.ndarray:
        vec = recon_volume.sum(axis=0).reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)


def load_longmemeval(path: Path, limit_questions: Optional[int] = None) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if limit_questions is not None:
        return data[:limit_questions]
    return data


def build_session_catalog(data: Sequence[dict]) -> Dict[str, SessionRecord]:
    catalog: Dict[str, SessionRecord] = {}
    for entry in data:
        for session, session_id, date in zip(
            entry["haystack_sessions"],
            entry["haystack_session_ids"],
            entry["haystack_dates"],
        ):
            if session_id in catalog:
                continue
            user_turns = [turn["content"] for turn in session if turn["role"] == "user"]
            text = "\n".join(user_turns).strip()
            catalog[session_id] = SessionRecord(session_id=session_id, date=date, text=text)
    return catalog


def build_aaak_docs(
    sessions: Dict[str, SessionRecord],
    vector_dim: int,
) -> Tuple[Dict[str, AAAKDoc], Dict[str, float], HashedTextVectorizer]:
    dialect = AAAKLiteDialect()
    vectorizer = HashedTextVectorizer(dim=vector_dim)
    compressed_texts = [
        dialect.compress(record.text, metadata={"date": record.date})
        for record in sessions.values()
    ]
    vectorizer.fit(compressed_texts)

    docs: Dict[str, AAAKDoc] = {}
    compress_times: List[float] = []
    vector_times: List[float] = []
    for record, compressed in zip(sessions.values(), compressed_texts):
        t0 = time.perf_counter()
        _ = compressed
        t1 = time.perf_counter()
        vector = vectorizer.transform(compressed)
        t2 = time.perf_counter()
        docs[record.session_id] = AAAKDoc(
            session_id=record.session_id,
            compressed_text=compressed,
            vector=vector,
            raw_bytes=len(record.text.encode("utf-8")),
            compressed_bytes=len(compressed.encode("utf-8")),
        )
        compress_times.append(t1 - t0)
        vector_times.append(t2 - t1)

    stats = {
        "avg_compress_seconds": float(np.mean(compress_times) if compress_times else 0.0),
        "avg_vectorize_seconds": float(np.mean(vector_times) if vector_times else 0.0),
    }
    return docs, stats, vectorizer


def build_k3m_docs(
    sessions: Dict[str, SessionRecord],
    projector_shape: Tuple[int, int, int],
) -> Tuple[Dict[str, K3MDoc], Dict[str, float], K3MTextProjector, K3MCodec]:
    projector = K3MTextProjector(shape=projector_shape)
    projector.fit([record.text for record in sessions.values()])
    codec = K3MCodec(
        base_radii=(1, 1, 1),
        tube_half_length=2,
        tube_radius=0,
        max_tubes=8,
        max_junctions=4,
        residual_budget=96,
        response_steps=2,
    )

    docs: Dict[str, K3MDoc] = {}
    encode_times: List[float] = []
    decode_times: List[float] = []
    for idx, record in enumerate(sessions.values(), start=1):
        volume = projector.text_to_volume(record.text)
        t0 = time.perf_counter()
        archive = codec.encode(volume)
        t1 = time.perf_counter()
        recon = codec.decode(archive)
        t2 = time.perf_counter()
        docs[record.session_id] = K3MDoc(
            session_id=record.session_id,
            archive=archive,
            vector=projector.doc_vector(recon),
            raw_bytes=len(record.text.encode("utf-8")),
            archive_bytes=archive.compressed_size_bytes(),
            encode_seconds=float(t1 - t0),
            decode_seconds=float(t2 - t1),
        )
        encode_times.append(t1 - t0)
        decode_times.append(t2 - t1)
        if idx % 2000 == 0:
            print(f"[k3m] encoded {idx}/{len(sessions)} sessions")

    stats = {
        "avg_encode_seconds": float(np.mean(encode_times) if encode_times else 0.0),
        "avg_decode_seconds": float(np.mean(decode_times) if decode_times else 0.0),
    }
    return docs, stats, projector, codec


def rank_candidates(
    question_vec: np.ndarray,
    candidate_ids: Sequence[str],
    doc_lookup: Dict[str, np.ndarray],
) -> List[str]:
    scored = []
    for session_id in candidate_ids:
        score = cosine_similarity(question_vec, doc_lookup[session_id])
        scored.append((score, session_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [session_id for _, session_id in scored]


def reciprocal_rank(ranked_ids: Sequence[str], answer_ids: set[str]) -> float:
    for idx, session_id in enumerate(ranked_ids, start=1):
        if session_id in answer_ids:
            return 1.0 / idx
    return 0.0


def best_rank(ranked_ids: Sequence[str], answer_ids: set[str]) -> Optional[int]:
    for idx, session_id in enumerate(ranked_ids, start=1):
        if session_id in answer_ids:
            return idx
    return None


def summarize_storage_aaak(docs: Dict[str, AAAKDoc]) -> Dict[str, float]:
    raw_bytes = np.array([doc.raw_bytes for doc in docs.values()], dtype=np.float64)
    compressed_bytes = np.array([doc.compressed_bytes for doc in docs.values()], dtype=np.float64)
    return {
        "avg_raw_bytes": float(raw_bytes.mean()),
        "avg_index_bytes": float(compressed_bytes.mean()),
        "avg_compression_ratio": float((raw_bytes / np.maximum(compressed_bytes, 1.0)).mean()),
    }


def summarize_storage_k3m(docs: Dict[str, K3MDoc]) -> Dict[str, float]:
    raw_bytes = np.array([doc.raw_bytes for doc in docs.values()], dtype=np.float64)
    archive_bytes = np.array([doc.archive_bytes for doc in docs.values()], dtype=np.float64)
    return {
        "avg_raw_bytes": float(raw_bytes.mean()),
        "avg_index_bytes": float(archive_bytes.mean()),
        "avg_compression_ratio": float((raw_bytes / np.maximum(archive_bytes, 1.0)).mean()),
    }


def evaluate_methods(
    data: Sequence[dict],
    aaak_docs: Dict[str, AAAKDoc],
    k3m_docs: Dict[str, K3MDoc],
    aaak_vectorizer: HashedTextVectorizer,
    projector: K3MTextProjector,
    codec: K3MCodec,
) -> Dict[str, object]:
    aaak_vectors = {session_id: doc.vector for session_id, doc in aaak_docs.items()}
    k3m_vectors = {session_id: doc.vector for session_id, doc in k3m_docs.items()}

    method_metrics = {
        "aaak": defaultdict(list),
        "k3m": defaultdict(list),
    }
    per_type = {
        "aaak": defaultdict(lambda: defaultdict(list)),
        "k3m": defaultdict(lambda: defaultdict(list)),
    }
    query_times = {"aaak": [], "k3m_rank": [], "k3m_query_decode": []}
    k3m_memory_query = MemoryQuery(direction=(1.0, 0.0, 0.0), time_range=None, modality="text")

    for idx, entry in enumerate(data, start=1):
        candidate_ids = entry["haystack_session_ids"]
        answer_ids = set(entry["answer_session_ids"])
        qtype = entry["question_type"]
        question = entry["question"]

        t0 = time.perf_counter()
        aaak_qvec = aaak_vectorizer.transform(question)
        aaak_ranked = rank_candidates(aaak_qvec, candidate_ids, aaak_vectors)
        t1 = time.perf_counter()
        k3m_qvec = projector.query_vector(question)
        k3m_ranked = rank_candidates(k3m_qvec, candidate_ids, k3m_vectors)
        t2 = time.perf_counter()
        top_k3m_id = k3m_ranked[0]
        codec.query_decode(k3m_docs[top_k3m_id].archive, k3m_memory_query, top_k=3)
        t3 = time.perf_counter()

        query_times["aaak"].append(t1 - t0)
        query_times["k3m_rank"].append(t2 - t1)
        query_times["k3m_query_decode"].append(t3 - t2)

        for method_name, ranked_ids in (("aaak", aaak_ranked), ("k3m", k3m_ranked)):
            rr = reciprocal_rank(ranked_ids, answer_ids)
            rank = best_rank(ranked_ids, answer_ids)
            method_metrics[method_name]["mrr"].append(rr)
            per_type[method_name][qtype]["mrr"].append(rr)
            for k in K_METRICS:
                hit = 1.0 if rank is not None and rank <= k else 0.0
                method_metrics[method_name][f"recall@{k}"].append(hit)
                per_type[method_name][qtype][f"recall@{k}"].append(hit)

        if idx % 100 == 0:
            print(f"[eval] processed {idx}/{len(data)} questions")

    overall = {}
    for method_name, metric_lists in method_metrics.items():
        overall[method_name] = {metric: float(np.mean(values)) for metric, values in metric_lists.items()}

    per_type_summary = {}
    for method_name, type_metrics in per_type.items():
        per_type_summary[method_name] = {}
        for qtype, metric_lists in type_metrics.items():
            per_type_summary[method_name][qtype] = {
                metric: float(np.mean(values)) for metric, values in metric_lists.items()
            }

    return {
        "overall": overall,
        "per_type": per_type_summary,
        "query_timing": {
            "aaak_query_seconds": float(np.mean(query_times["aaak"])),
            "k3m_rank_seconds": float(np.mean(query_times["k3m_rank"])),
            "k3m_query_decode_seconds": float(np.mean(query_times["k3m_query_decode"])),
        },
    }


def generate_markdown_report(
    dataset_stats: Dict[str, object],
    storage_aaak: Dict[str, float],
    storage_k3m: Dict[str, float],
    aaak_build_stats: Dict[str, float],
    k3m_build_stats: Dict[str, float],
    evaluation: Dict[str, object],
    projector_shape: Tuple[int, int, int],
) -> str:
    overall = evaluation["overall"]
    per_type = evaluation["per_type"]
    timing = evaluation["query_timing"]

    lines = [
        "# K3M vs AAAK Benchmark Compare Report",
        "",
        "## Setup",
        f"- Dataset: LongMemEval cleaned (`{dataset_stats['questions']}` questions, `{dataset_stats['unique_sessions']}` unique haystack sessions)",
        "- Corpus granularity: session-level, using concatenated user turns per session",
        "- AAAK side: MemPalace-style lossy structured summary, then raw-question retrieval over compressed text",
        "- K3M side: text projected into a 3D hashed memory volume, encoded by `K3MCodec`, then retrieved from reconstructed compressed memory",
        f"- K3M projection shape: `{projector_shape}`",
        "",
        "## Overall Comparison",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Avg bytes/session | Avg compression | Avg build encode | Avg query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| AAAK | {overall['aaak']['recall@1']:.4f} | {overall['aaak']['recall@3']:.4f} | {overall['aaak']['recall@5']:.4f} | {overall['aaak']['recall@10']:.4f} | {overall['aaak']['mrr']:.4f} | {storage_aaak['avg_index_bytes']:.1f} | {storage_aaak['avg_compression_ratio']:.2f}x | {(aaak_build_stats['avg_compress_seconds'] + aaak_build_stats['avg_vectorize_seconds']):.4f}s | {timing['aaak_query_seconds']:.4f}s |",
        f"| K3M | {overall['k3m']['recall@1']:.4f} | {overall['k3m']['recall@3']:.4f} | {overall['k3m']['recall@5']:.4f} | {overall['k3m']['recall@10']:.4f} | {overall['k3m']['mrr']:.4f} | {storage_k3m['avg_index_bytes']:.1f} | {storage_k3m['avg_compression_ratio']:.2f}x | {(k3m_build_stats['avg_encode_seconds'] + k3m_build_stats['avg_decode_seconds']):.4f}s | {(timing['k3m_rank_seconds'] + timing['k3m_query_decode_seconds']):.4f}s |",
        "",
        "## Per Question Type",
        "",
        "| Question type | AAAK R@5 | K3M R@5 | AAAK MRR | K3M MRR |",
        "|---|---:|---:|---:|---:|",
    ]

    for qtype in dataset_stats["question_types"]:
        lines.append(
            f"| {qtype} | {per_type['aaak'][qtype]['recall@5']:.4f} | {per_type['k3m'][qtype]['recall@5']:.4f} | {per_type['aaak'][qtype]['mrr']:.4f} | {per_type['k3m'][qtype]['mrr']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Storage Notes",
            f"- AAAK average raw text bytes per session: {storage_aaak['avg_raw_bytes']:.1f}",
            f"- AAAK average compressed text bytes per session: {storage_aaak['avg_index_bytes']:.1f}",
            f"- K3M average raw text bytes per session: {storage_k3m['avg_raw_bytes']:.1f}",
            f"- K3M average archive bytes per session: {storage_k3m['avg_index_bytes']:.1f}",
            "",
            "## Timing Notes",
            f"- AAAK average index build time/session: {(aaak_build_stats['avg_compress_seconds'] + aaak_build_stats['avg_vectorize_seconds']):.4f}s",
            f"- K3M average encode+decode time/session: {(k3m_build_stats['avg_encode_seconds'] + k3m_build_stats['avg_decode_seconds']):.4f}s",
            f"- AAAK average query ranking time/question: {timing['aaak_query_seconds']:.4f}s",
            f"- K3M average query ranking time/question: {timing['k3m_rank_seconds']:.4f}s",
            f"- K3M average local `query_decode()` time/question: {timing['k3m_query_decode_seconds']:.4f}s",
            "",
            "## Caveats",
            "- This AAAK path is a MemPalace-style reproduction: lossy structured summary plus retrieval over compressed text, but it does not invoke ChromaDB or the original MemPal benchmark runner.",
            "- This K3M path evaluates retrieval after compressing hashed 3D text memory volumes, not after plugging K3M into a transformer memory backend.",
            "- The result should be interpreted as an apples-to-apples benchmark on the same LongMemEval haystacks and answers, not as a claim that it exactly reproduces MemPalace's published numbers.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark K3M against AAAK-style compression on LongMemEval.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/longmemeval_s_cleaned.json"),
        help="Path to LongMemEval cleaned JSON",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="Optionally benchmark only the first N questions",
    )
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=2048,
        help="AAAK hashed vector dimension",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=(12, 12, 12),
        metavar=("T", "U", "V"),
        help="K3M text-volume shape",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("benchmark_compare_report.md"),
        help="Markdown report output path",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_compare_results.json"),
        help="JSON result output path",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    data = load_longmemeval(args.data, limit_questions=args.limit_questions)
    sessions = build_session_catalog(data)
    question_type_counts = Counter(entry["question_type"] for entry in data)

    print(f"[setup] loaded {len(data)} questions and {len(sessions)} unique sessions")
    aaak_docs, aaak_build_stats, aaak_vectorizer = build_aaak_docs(sessions, vector_dim=args.vector_dim)
    print("[setup] finished AAAK document build")
    k3m_docs, k3m_build_stats, projector, codec = build_k3m_docs(
        sessions,
        projector_shape=tuple(args.shape),
    )
    print("[setup] finished K3M document build")

    storage_aaak = summarize_storage_aaak(aaak_docs)
    storage_k3m = summarize_storage_k3m(k3m_docs)
    evaluation = evaluate_methods(
        data=data,
        aaak_docs=aaak_docs,
        k3m_docs=k3m_docs,
        aaak_vectorizer=aaak_vectorizer,
        projector=projector,
        codec=codec,
    )

    dataset_stats = {
        "questions": len(data),
        "unique_sessions": len(sessions),
        "question_types": [name for name, _ in question_type_counts.most_common()],
    }

    report = generate_markdown_report(
        dataset_stats=dataset_stats,
        storage_aaak=storage_aaak,
        storage_k3m=storage_k3m,
        aaak_build_stats=aaak_build_stats,
        k3m_build_stats=k3m_build_stats,
        evaluation=evaluation,
        projector_shape=tuple(args.shape),
    )

    result_payload = {
        "dataset": dataset_stats,
        "storage": {"aaak": storage_aaak, "k3m": storage_k3m},
        "build_stats": {"aaak": aaak_build_stats, "k3m": k3m_build_stats},
        "evaluation": evaluation,
        "elapsed_seconds": float(time.perf_counter() - start),
    }

    args.report_out.write_text(report, encoding="utf-8")
    args.json_out.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    print(report)
    print(f"[done] wrote {args.report_out}")
    print(f"[done] wrote {args.json_out}")


if __name__ == "__main__":
    main()
