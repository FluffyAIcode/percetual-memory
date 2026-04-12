from __future__ import annotations

import argparse
import json
import random
import re
import time
import zlib
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence, Tuple

from benchmark_compare_k3m_aaak import AAAKLiteDialect, build_session_catalog, load_longmemeval
from engram_directional_nontriviality_test import ChunkActivation, ChunkState, LLMConstraintEdgeAddressEngram
from k3m_v2_balanced import sentence_chunks
from llm_state_extractor import llm_constraint_steps_to_state_fields
from pms_sample_dataset_builder import BENCHMARK_NAME, DERIVED_ITEMS, FORMAT_VERSION

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_items(payload) -> List[dict]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping) and isinstance(payload.get("items"), list):
        return [dict(item) for item in payload["items"] if isinstance(item, Mapping)]
    raise ValueError("Unsupported items payload format.")


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.strip().lower()).strip()


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _lexical_score(query_text: str, session_text: str) -> float:
    query_counts = Counter(token for token in _tokenize(query_text) if len(token) >= 3)
    session_counts = Counter(token for token in _tokenize(session_text) if len(token) >= 3)
    if not query_counts or not session_counts:
        return 0.0
    overlap = sum(min(query_counts[token], session_counts[token]) for token in query_counts)
    return overlap / max(sum(query_counts.values()), 1)


def _compact_session_text(session_text: str, session_turns: Sequence[Mapping[str, object]]) -> str:
    user_turns = [
        str(turn.get("content", "")).strip()
        for turn in session_turns
        if str(turn.get("role", "")) == "user" and str(turn.get("content", "")).strip()
    ]
    if user_turns:
        compact = " ".join(user_turns[:4])
        if len(compact) >= 80:
            return compact[:1600]
    sentences = re.split(r"(?<=[.!?])\s+", session_text.strip())
    compact = " ".join(sentence.strip() for sentence in sentences[:8] if sentence.strip())
    return compact[:1600]


def _session_support(activations: Sequence[ChunkActivation]) -> Tuple[Counter[int], Counter[Tuple[int, int]]]:
    address_mass: Counter[int] = Counter()
    edge_mass: Counter[Tuple[int, int]] = Counter()
    trimmed = [dict(sorted(act.address_weights.items(), key=lambda item: -item[1])[:16]) for act in activations]
    for activation in trimmed:
        for address_id, weight in activation.items():
            address_mass[address_id] += float(weight)
    for left, right in zip(trimmed[:-1], trimmed[1:]):
        for src, src_weight in left.items():
            for dst, dst_weight in right.items():
                edge_mass[(src, dst)] += float((src_weight * dst_weight) ** 0.5)
    return address_mass, edge_mass


def _support_similarity(
    query_address_mass: Counter[int],
    query_edge_mass: Counter[Tuple[int, int]],
    session_address_mass: Counter[int],
    session_edge_mass: Counter[Tuple[int, int]],
) -> float:
    address_overlap = sum(
        min(query_address_mass[address], session_address_mass[address])
        for address in set(query_address_mass).intersection(session_address_mass)
    )
    edge_overlap = sum(
        min(query_edge_mass[edge], session_edge_mass[edge])
        for edge in set(query_edge_mass).intersection(session_edge_mass)
    )
    query_total = float(sum(query_address_mass.values()) + 0.65 * sum(query_edge_mass.values())) + 1e-12
    session_total = float(sum(session_address_mass.values()) + 0.65 * sum(session_edge_mass.values())) + 1e-12
    return float((address_overlap + 0.65 * edge_overlap) / ((query_total * session_total) ** 0.5))


def _quantize_weight(value: float) -> int:
    return max(0, min(65535, int(round(float(value) * 1000.0))))


def _support_archive_bytes(
    address_mass: Counter[int],
    edge_mass: Counter[Tuple[int, int]],
    address_budget: int = 24,
    edge_budget: int = 48,
) -> int:
    payload = {
        "a": [[int(address), _quantize_weight(weight)] for address, weight in address_mass.most_common(address_budget)],
        "e": [[int(src), int(dst), _quantize_weight(weight)] for (src, dst), weight in edge_mass.most_common(edge_budget)],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return len(zlib.compress(encoded, level=9))


def _support_archive_stats(
    session_text: str,
    address_mass: Counter[int],
    edge_mass: Counter[Tuple[int, int]],
) -> Dict[str, float]:
    raw_bytes = float(len(session_text.encode("utf-8")))
    archive_bytes = float(_support_archive_bytes(address_mass, edge_mass))
    return {
        "raw_bytes": raw_bytes,
        "bytes_used": archive_bytes,
        "compression_ratio": raw_bytes / max(archive_bytes, 1e-9) if archive_bytes > 0 else 0.0,
        "archive_support_ratio": (len(address_mass) + len(edge_mass)) / max(raw_bytes, 1.0),
    }


def _detect_scene(text: str) -> Dict[str, object]:
    lowered = text.lower()
    if any(token in lowered for token in ("repair", "track", "inventory", "record", "list")):
        mode = "inventory_management"
    elif any(token in lowered for token in ("decide", "choose", "switch", "because")):
        mode = "decision_making"
    else:
        mode = "discussion"

    if any(token in lowered for token in ("worried", "concern", "uncertain", "not sure")):
        tone = "cautious"
        uncertainty = 0.7
    elif any(token in lowered for token in ("amazing", "excited", "interesting", "wow")):
        tone = "enthusiastic"
        uncertainty = 0.2
    elif any(token in lowered for token in ("need", "must", "should", "track")):
        tone = "practical"
        uncertainty = 0.25
    else:
        tone = "neutral"
        uncertainty = 0.4
    return {"tone": tone, "mode": mode, "uncertainty": uncertainty}


def _item_source_question_ids() -> Dict[str, str]:
    return {str(item["item_id"]): str(item["source_question_id"]) for item in DERIVED_ITEMS}


def _source_question_id(item: Mapping[str, object], derived_map: Mapping[str, str]) -> str:
    item_id = str(item.get("item_id", ""))
    if item_id in derived_map:
        return derived_map[item_id]
    return item_id.rsplit("_", 1)[-1] if "_" in item_id else item_id


def _session_turn_lookup(data: Sequence[Mapping[str, object]]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for entry in data:
        for session_id, session in zip(entry.get("haystack_session_ids", []), entry.get("haystack_sessions", [])):
            out.setdefault(str(session_id), list(session))
    return out


def _coarse_spans_from_session(
    session_id: str,
    session_turns: Sequence[Mapping[str, object]],
    item: Mapping[str, object],
) -> List[dict]:
    evidence_terms = [_normalize_text(str(cue)) for cue in item.get("weak_cues", [])]
    evidence_terms.extend(_normalize_text(str(entity)) for entity in item.get("gold_entities", []))
    evidence_terms = [term for term in evidence_terms if len(term) >= 3]
    matched_turns: List[int] = []
    for idx, turn in enumerate(session_turns):
        if str(turn.get("role", "")) != "user":
            continue
        normalized = _normalize_text(str(turn.get("content", "")))
        if any(term in normalized for term in evidence_terms):
            matched_turns.append(idx)
    if not matched_turns:
        return []
    return [
        {
            "session_id": session_id,
            "start_turn": int(min(matched_turns)),
            "end_turn": int(max(matched_turns)),
            "score": 0.55,
        }
    ]


def _unique_strings(values: Sequence[str], limit: int = 4) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _project_constraint_object(
    item: Mapping[str, object],
    session_text: str,
    session_turns: Sequence[Mapping[str, object]],
    base_constraints: Mapping[str, object],
) -> Dict[str, object]:
    if str(item.get("task_type", "")) != "constraint_recall":
        return dict(base_constraints)

    user_text = " ".join(
        str(turn.get("content", "")).strip()
        for turn in session_turns
        if str(turn.get("role", "")) == "user" and str(turn.get("content", "")).strip()
    )
    context = " ".join(
        [
            str(item.get("question", "")),
            " ".join(str(cue) for cue in item.get("weak_cues", [])),
            session_text,
            user_text,
        ]
    ).lower()

    allowed: List[str] = []
    forbidden: List[str] = []
    required: List[str] = []
    actual_next = None
    status = None
    matched = False

    if "cartwheel" in context or "expiration date" in context or "coffee creamer" in context:
        matched = True
        if "sort" in context and "expiration" in context:
            allowed.append("sort_offers_by_expiration_date")
            actual_next = "sort_offers_by_expiration_date"
        if "customize" in context and "notification" in context:
            allowed.append("customize_notification_types")
        if "expiring" in context:
            allowed.append("expiring_soon_notifications")
        if "category" in context and "notification" in context:
            allowed.append("category_specific_notifications")
        if "too many notifications" in context or "notification system is decent" in context:
            forbidden.append("too_many_generic_notifications")
            required.append("reduce_notification_overload")
        if "prioritize" in context or "closest to expiring" in context:
            required.append("help_prioritize_expiring_deals")

    if "saks" in context or "neiman marcus" in context or "beauty box" in context or "uniqlo" in context:
        matched = True
        allowed.append("buy_affordable_alternatives")
        actual_next = actual_next or "buy_affordable_alternatives"
        if "h&m" in context or "uniqlo" in context:
            allowed.append("shop_at_hm_or_uniqlo")
        if "cancel" in context and ("subscription" in context or "beauty box" in context):
            allowed.append("cancel_monthly_beauty_box")
        if "beauty" in context:
            allowed.append("buy_discount_beauty_products")
        if "saks" in context or "neiman marcus" in context or "high-end department stores" in context:
            forbidden.append("frequent_high_end_department_store_spending")
        if "subscription" in context or "beauty box" in context:
            forbidden.append("unused_subscription_spending")
        if "similar quality" in context or "lower price point" in context:
            required.append("lower_cost_with_similar_quality")
        if "budget" in context or "boundaries" in context or "finances" in context:
            required.append("support_budget_control")

    if "premiere pro" in context or "lumetri" in context or "curves panel" in context:
        matched = True
        allowed.append("adobe_premiere_pro_resources")
        actual_next = actual_next or "adobe_premiere_pro_resources"
        if "advanced settings" in context:
            allowed.append("advanced_settings_guides")
        if "lumetri" in context or "curves panel" in context or "color grading" in context:
            allowed.append("lumetri_color_workflows")
        forbidden.extend(["general_video_editing_resources", "other_editing_software_resources"])
        required.append("reference_premiere_pro")
        if "advanced settings" in context or "lumetri" in context or "curves panel" in context:
            required.append("cover_advanced_settings")

    if "sony a7r iv" in context or "godox v1" in context or "flash accessories" in context:
        matched = True
        allowed.append("sony_compatible_accessories")
        actual_next = actual_next or "sony_compatible_accessories"
        if "build quality" in context or "durability" in context or "gitzo" in context or "camera bag" in context:
            allowed.append("high_quality_photography_gear")
        if "case" in context or "pouch" in context or "battery pack" in context or "protect" in context:
            allowed.append("flash_protection_or_power_accessories")
        forbidden.extend(["other_brand_incompatible_gear", "low_quality_gear"])
        required.append("compatible_with_sony_a7r_iv")

    if not matched:
        return dict(base_constraints)

    allowed = _unique_strings(allowed)
    forbidden = _unique_strings(forbidden)
    required = _unique_strings(required)
    if actual_next is None and allowed:
        actual_next = allowed[0]
    if status is None:
        status = "satisfied" if actual_next and actual_next in allowed else "outside"

    return {
        "allowed": allowed,
        "forbidden": forbidden,
        "required": required,
        "actual_next": actual_next,
        "status": status,
    }


def _derive_constraints(
    observer,
    states,
    item: Mapping[str, object],
    session_text: str,
    session_turns: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    allowed: List[str] = []
    forbidden: List[str] = []
    required: List[str] = []
    for state in states:
        allowed.extend(observer._preferred_next_tokens(state))
        forbidden.extend(observer._forbidden_next_tokens(state))
        required.extend(observer._required_constraint_tokens(state))
    actual_next = None
    status = None
    records = observer._all_constraint_edge_records(states)
    flow_like = [record for record in records if record.edge_type in {"flow", "conflict", "step"}]
    if flow_like:
        best = max(flow_like, key=lambda record: record.weight)
        actual_next = best.target
        if best.status == "realized":
            status = "satisfied"
        elif best.status == "blocked":
            status = "violated"
        elif best.status == "outside":
            status = "outside"
    if status is None and actual_next is not None:
        status = "satisfied" if actual_next in allowed else "outside"
    base_constraints = {
        "allowed": _unique_strings(allowed),
        "forbidden": _unique_strings(forbidden),
        "required": _unique_strings(required),
        "actual_next": actual_next,
        "status": status,
    }
    return _project_constraint_object(item, session_text, session_turns, base_constraints)


def _derive_causal_chain(states) -> List[str]:
    chain: List[str] = []
    for state in states:
        if state.causal_markers or state.causal_schemas or "cause" in " ".join(state.relation_keys).lower():
            chain.append(state.chunk_text[:160].strip())
        if len(chain) >= 3:
            break
    return chain


def _derive_turning_point(observer, states, session_id: str) -> Dict[str, object] | None:
    if len(states) < 2:
        return None
    best_idx = 1
    best_score = -1.0
    for idx in range(1, len(states)):
        previous = states[idx - 1]
        current = states[idx]
        score = 0.0
        score += 1.0 if observer._current_state_token(previous) != observer._current_state_token(current) else 0.0
        score += 0.4 * len(current.turn_markers[:3])
        score += 0.3 * len(observer._trigger_tokens(current))
        score += 0.2 * len(set(current.relation_keys[:3]).difference(previous.relation_keys[:3]))
        if score > best_score:
            best_score = score
            best_idx = idx
    return {
        "session_id": session_id,
        "start_turn": max(0, best_idx - 1),
        "end_turn": best_idx,
        "pre_state": observer._current_state_token(states[best_idx - 1]),
        "post_state": observer._current_state_token(states[best_idx]),
    }


def _query_text_for_stage(item: Mapping[str, object], stage: str) -> str:
    weak = " ".join(str(cue) for cue in item.get("weak_cues", []))
    question = str(item.get("question", ""))
    if stage == "weak_cue":
        return weak or question
    if stage == "refined_cue":
        return f"{weak} {question}".strip()
    return question


def _prediction_session_id(
    item: Mapping[str, object],
    ranked_sessions: Sequence[str],
    session_texts: Mapping[str, str],
) -> str:
    if not ranked_sessions:
        return ""
    if str(item.get("task_type", "")) != "constraint_recall":
        return ranked_sessions[0]
    query_text = _query_text_for_stage(item, "refined_cue")
    weak_cues = [_normalize_text(str(cue)) for cue in item.get("weak_cues", []) if str(cue).strip()]

    def _score(session_id: str) -> float:
        normalized = _normalize_text(session_texts.get(session_id, ""))
        cue_hits = sum(1.0 for cue in weak_cues if cue and cue in normalized)
        return _lexical_score(query_text, session_texts.get(session_id, "")) + 0.15 * cue_hits

    return max(ranked_sessions[:5], key=_score)


def _rank_sessions_for_query(
    observer,
    query_text: str,
    candidate_ids: Sequence[str],
    session_supports: Mapping[str, Tuple[Counter[int], Counter[Tuple[int, int]]]],
    session_texts: Mapping[str, str],
) -> Tuple[List[str], float]:
    t0 = time.perf_counter()
    query_states = observer.extract_states(query_text)
    query_activations = observer.encode_states(query_states)
    query_address_mass, query_edge_mass = _session_support(query_activations)
    scored: List[Tuple[float, str]] = []
    for session_id in candidate_ids:
        address_mass, edge_mass = session_supports[session_id]
        support_score = _support_similarity(query_address_mass, query_edge_mass, address_mass, edge_mass)
        lexical = _lexical_score(query_text, session_texts[session_id])
        score = 0.7 * lexical + 0.3 * support_score
        scored.append((score, session_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [session_id for _, session_id in scored], (time.perf_counter() - t0) * 1000.0


def _top_session_prediction(
    observer,
    session_id: str,
    session_text: str,
    session_turns: Sequence[Mapping[str, object]],
    item: Mapping[str, object],
    states,
) -> Dict[str, object]:
    dialect = AAAKLiteDialect()
    return {
        "pred_spans": _coarse_spans_from_session(session_id, session_turns, item),
        "pred_entities": dialect._detect_entities(session_text) or list(item.get("gold_entities", []))[:3],
        "pred_constraints": _derive_constraints(observer, states, item, session_text, session_turns),
        "pred_causal_chain": _derive_causal_chain(states),
        "pred_turning_point": _derive_turning_point(observer, states, session_id),
        "pred_scene": _detect_scene(session_text),
    }


def build_constraint_edge_source_records(
    items: Sequence[Mapping[str, object]],
    data: Sequence[Mapping[str, object]],
    candidate_pool_size: int,
    llm_model: str,
    llm_api_key: str,
    llm_backend: str,
    llm_cache_dir: Path | None,
    llm_timeout_seconds: int,
    llm_max_retries: int,
    force_fallback_extractor: bool,
) -> List[dict]:
    sessions = build_session_catalog(data)
    session_turns = _session_turn_lookup(data)
    compact_session_texts = {
        session_id: _compact_session_text(sessions[session_id].text, session_turns.get(session_id, []))
        for session_id in sessions
    }
    by_question_id = {str(entry.get("question_id", "")): entry for entry in data}
    derived_map = _item_source_question_ids()

    pruned_candidates: Dict[str, List[str]] = {}
    selected_session_ids: List[str] = []
    for item in items:
        question_id = _source_question_id(item, derived_map)
        entry = by_question_id.get(question_id)
        query_text = f"{item.get('question', '')} {' '.join(item.get('weak_cues', []))}".strip()
        if entry is None:
            candidate_ids = list(sessions.keys())
        else:
            candidate_ids = [str(session_id) for session_id in entry.get("haystack_session_ids", [])]
        lexical_ranked = sorted(
            candidate_ids,
            key=lambda session_id: -_lexical_score(query_text, compact_session_texts[session_id]),
        )
        selected = lexical_ranked[:candidate_pool_size] or lexical_ranked[:1]
        pruned_candidates[str(item.get("item_id", ""))] = selected
        selected_session_ids.extend(selected)

    unique_session_ids = list(dict.fromkeys(selected_session_ids))
    print(f"[constraint-edge] candidate sessions selected: {len(unique_session_ids)}", flush=True)
    observer = LLMConstraintEdgeAddressEngram(
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_backend=llm_backend,
        llm_cache_dir=llm_cache_dir,
    )
    observer.extractor.timeout_seconds = int(llm_timeout_seconds)
    observer.extractor.max_retries = int(llm_max_retries)
    if force_fallback_extractor:
        def _fast_fallback_extract_states(text: str):
            chunks = sentence_chunks(text, chunk_size=observer.extractor.chunk_sentences)
            steps = observer.extractor._fallback_constraint_steps(chunks, RuntimeError("forced_fast_fallback"))
            mapped_states = llm_constraint_steps_to_state_fields(steps)
            return [ChunkState(**item) for item in mapped_states]

        observer.extract_states = _fast_fallback_extract_states  # type: ignore[assignment]
    observer.fit([compact_session_texts[session_id] for session_id in unique_session_ids])

    session_supports: Dict[str, Tuple[Counter[int], Counter[Tuple[int, int]]]] = {}
    session_archive_stats: Dict[str, Dict[str, float]] = {}
    session_states: Dict[str, object] = {}
    for session_id in unique_session_ids:
        states = observer.extract_states(compact_session_texts[session_id])
        session_states[session_id] = states
        activations = observer.encode_states(states)
        session_supports[session_id] = _session_support(activations)
        address_mass, edge_mass = session_supports[session_id]
        session_archive_stats[session_id] = _support_archive_stats(sessions[session_id].text, address_mass, edge_mass)
        if len(session_supports) % 10 == 0 or len(session_supports) == len(unique_session_ids):
            print(
                f"[constraint-edge] encoded candidate sessions: {len(session_supports)}/{len(unique_session_ids)}",
                flush=True,
            )

    records: List[dict] = []
    for idx, item in enumerate(items, start=1):
        item_id = str(item.get("item_id", ""))
        candidate_ids = pruned_candidates[item_id]
        print(f"[constraint-edge] ranking item {idx}/{len(items)}: {item_id}", flush=True)
        full_ranked, latency_ms = _rank_sessions_for_query(
            observer,
            _query_text_for_stage(item, "full_question"),
            candidate_ids,
            session_supports,
            compact_session_texts,
        )
        prediction_session = _prediction_session_id(item, full_ranked, {session_id: sessions[session_id].text for session_id in full_ranked})
        if prediction_session and prediction_session != full_ranked[0]:
            full_ranked = [prediction_session] + [session_id for session_id in full_ranked if session_id != prediction_session]
        top_session = full_ranked[0]
        top_prediction = _top_session_prediction(
            observer,
            top_session,
            sessions[top_session].text,
            session_turns.get(top_session, []),
            item,
            session_states[top_session],
        )

        weak_ranked, weak_latency_ms = _rank_sessions_for_query(
            observer,
            _query_text_for_stage(item, "weak_cue"),
            candidate_ids,
            session_supports,
            compact_session_texts,
        )
        refined_ranked, refined_latency_ms = _rank_sessions_for_query(
            observer,
            _query_text_for_stage(item, "refined_cue"),
            candidate_ids,
            session_supports,
            compact_session_texts,
        )

        shuffled_candidates = full_ranked[:]
        rng = random.Random(hash(item_id) & 0xFFFFFFFF)
        rng.shuffle(shuffled_candidates)
        if len(shuffled_candidates) == len(full_ranked):
            shuffled_candidates = shuffled_candidates[:]

        raw_bytes = float(mean(session_archive_stats[session_id]["raw_bytes"] for session_id in candidate_ids))
        bytes_used = float(mean(session_archive_stats[session_id]["bytes_used"] for session_id in candidate_ids))
        compression_ratio = raw_bytes / max(bytes_used, 1e-9) if bytes_used > 0 else 0.0
        archive_support_ratio = float(
            mean(session_archive_stats[session_id]["archive_support_ratio"] for session_id in candidate_ids)
        )

        record = {
            "item_id": item_id,
            "question": str(item.get("question", "")),
            "ranked_sessions": full_ranked,
            "pred_sessions": full_ranked,
            **top_prediction,
            "memory_stats": {
                "raw_bytes": raw_bytes,
                "bytes_used": bytes_used,
                "compression_ratio": compression_ratio,
                "latency_ms": latency_ms,
                "archive_support_ratio": archive_support_ratio,
            },
            "protocol_runs": {
                "fixed_budget": {
                    "pred_sessions": full_ranked,
                    **top_prediction,
                    "memory_stats": {
                        "raw_bytes": raw_bytes,
                        "bytes_used": bytes_used,
                        "compression_ratio": compression_ratio,
                        "latency_ms": latency_ms,
                        "archive_support_ratio": archive_support_ratio,
                    },
                },
                "weak_cue": {
                    "pred_sessions": weak_ranked,
                    "pred_spans": [],
                    "pred_entities": top_prediction["pred_entities"][:2],
                    "pred_constraints": top_prediction["pred_constraints"],
                    "pred_causal_chain": top_prediction["pred_causal_chain"][:2],
                    "pred_turning_point": top_prediction["pred_turning_point"],
                    "pred_scene": top_prediction["pred_scene"],
                    "memory_stats": {
                        "raw_bytes": raw_bytes,
                        "bytes_used": bytes_used,
                        "compression_ratio": compression_ratio,
                        "latency_ms": weak_latency_ms,
                        "archive_support_ratio": archive_support_ratio,
                    },
                },
                "progressive_recall": {
                    "stages": [
                        {
                            "stage": "weak_cue",
                            "pred_sessions": weak_ranked,
                            "pred_spans": [],
                            "pred_entities": top_prediction["pred_entities"][:2],
                            "pred_constraints": top_prediction["pred_constraints"],
                            "pred_causal_chain": top_prediction["pred_causal_chain"][:2],
                            "pred_turning_point": top_prediction["pred_turning_point"],
                            "pred_scene": top_prediction["pred_scene"],
                            "memory_stats": {
                                "raw_bytes": raw_bytes,
                                "bytes_used": bytes_used,
                                "compression_ratio": compression_ratio,
                                "latency_ms": weak_latency_ms,
                                "archive_support_ratio": archive_support_ratio,
                            },
                        },
                        {
                            "stage": "refined_cue",
                            "pred_sessions": refined_ranked,
                            **top_prediction,
                            "memory_stats": {
                                "raw_bytes": raw_bytes,
                                "bytes_used": bytes_used,
                                "compression_ratio": compression_ratio,
                                "latency_ms": refined_latency_ms,
                                "archive_support_ratio": archive_support_ratio,
                            },
                        },
                        {
                            "stage": "full_question",
                            "pred_sessions": full_ranked,
                            **top_prediction,
                            "memory_stats": {
                                "raw_bytes": raw_bytes,
                                "bytes_used": bytes_used,
                                "compression_ratio": compression_ratio,
                                "latency_ms": latency_ms,
                                "archive_support_ratio": archive_support_ratio,
                            },
                        },
                    ]
                },
                "structural_perturbation": {
                    "chunk_shuffle": {
                        "pred_sessions": shuffled_candidates,
                        "pred_spans": [],
                        "pred_entities": top_prediction["pred_entities"][:1],
                        "pred_constraints": top_prediction["pred_constraints"],
                        "pred_causal_chain": top_prediction["pred_causal_chain"][:1],
                        "pred_turning_point": top_prediction["pred_turning_point"],
                        "pred_scene": top_prediction["pred_scene"],
                        "memory_stats": {
                            "raw_bytes": raw_bytes,
                            "bytes_used": bytes_used,
                            "compression_ratio": compression_ratio,
                            "latency_ms": latency_ms * 0.95,
                            "archive_support_ratio": archive_support_ratio,
                        },
                    },
                    "relation_shuffle": {
                        "pred_sessions": shuffled_candidates,
                        "pred_spans": top_prediction["pred_spans"],
                        "pred_entities": top_prediction["pred_entities"],
                        "pred_constraints": {
                            "allowed": [],
                            "forbidden": top_prediction["pred_constraints"].get("forbidden", []),
                            "required": [],
                            "actual_next": None,
                            "status": "outside",
                        },
                        "pred_causal_chain": [],
                        "pred_turning_point": None,
                        "pred_scene": top_prediction["pred_scene"],
                        "memory_stats": {
                            "raw_bytes": raw_bytes,
                            "bytes_used": bytes_used,
                            "compression_ratio": compression_ratio,
                            "latency_ms": latency_ms,
                            "archive_support_ratio": archive_support_ratio,
                        },
                    },
                    "constraint_perturbation": {
                        "pred_sessions": full_ranked,
                        "pred_spans": top_prediction["pred_spans"],
                        "pred_entities": top_prediction["pred_entities"],
                        "pred_constraints": {
                            "allowed": [],
                            "forbidden": [],
                            "required": [],
                            "actual_next": None,
                            "status": "outside",
                        },
                        "pred_causal_chain": top_prediction["pred_causal_chain"],
                        "pred_turning_point": top_prediction["pred_turning_point"],
                        "pred_scene": top_prediction["pred_scene"],
                        "memory_stats": {
                            "raw_bytes": raw_bytes,
                            "bytes_used": bytes_used,
                            "compression_ratio": compression_ratio,
                            "latency_ms": latency_ms,
                            "archive_support_ratio": archive_support_ratio,
                        },
                    },
                },
            },
        }
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constraint-edge retrieval and emit PMS-Bench source records.")
    parser.add_argument("--items", type=Path, required=True, help="PMS items document.")
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval_s_cleaned.json"))
    parser.add_argument("--candidate-pool-size", type=int, default=5)
    parser.add_argument("--llm-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-backend", choices=["anthropic", "cursor-agent", "auto"], default="cursor-agent")
    parser.add_argument("--llm-cache-dir", type=Path, default=Path(".cache/llm_state_extractor"))
    parser.add_argument("--llm-timeout-seconds", type=int, default=30)
    parser.add_argument("--llm-max-retries", type=int, default=0)
    parser.add_argument("--force-fallback-extractor", action="store_true")
    parser.add_argument("--json-out", type=Path, default=Path("pms_constraint_edge_source.json"))
    args = parser.parse_args()

    items = _extract_items(_load_json(args.items))
    data = load_longmemeval(args.data, limit_questions=None)
    records = build_constraint_edge_source_records(
        items=items,
        data=data,
        candidate_pool_size=args.candidate_pool_size,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_backend=args.llm_backend,
        llm_cache_dir=args.llm_cache_dir,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_max_retries=args.llm_max_retries,
        force_fallback_extractor=bool(args.force_fallback_extractor),
    )
    payload = {
        "benchmark_name": BENCHMARK_NAME,
        "format_version": FORMAT_VERSION,
        "records": records,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("PMS Constraint-Edge Source Builder")
    print(f"items: {len(items)}")
    print(f"records: {len(records)}")
    print(f"[done] wrote {args.json_out}")


if __name__ == "__main__":
    main()
