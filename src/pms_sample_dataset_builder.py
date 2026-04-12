from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from benchmark_compare_k3m_aaak import AAAKLiteDialect, load_longmemeval

BENCHMARK_NAME = "PMS-Bench"
FORMAT_VERSION = "2.0"


CURATED_ITEM_OVERRIDES: Dict[str, Dict[str, object]] = {
    "e47becba": {
        "weak_cues": ["new job", "paperwork", "expense tracking"],
        "gold_entities": ["Business Administration"],
    },
    "51a45a95": {
        "weak_cues": ["Cartwheel app", "coffee creamer", "email inbox"],
        "gold_entities": ["Target", "Cartwheel"],
    },
    "58bf7951": {
        "weak_cues": ["community theater", "lead actress", "Emily audition"],
        "gold_entities": ["The Glass Menagerie", "Emily"],
    },
    "1e043500": {
        "weak_cues": ["ambient and lo-fi", "music and podcasts", "guitar practice"],
        "gold_entities": ["Summer Vibes", "Spotify"],
    },
    "f8c5f88b": {
        "weak_cues": ["new tennis racket", "sports shopping", "downtown"],
        "gold_entities": ["sports store downtown"],
    },
    "8a2466db": {
        "weak_cues": ["Premiere Pro", "Lumetri Color", "Curves panel"],
        "gold_entities": ["Adobe Premiere Pro", "Lumetri Color", "Curves panel"],
        "gold_constraints": {
            "allowed": [
                "adobe_premiere_pro_resources",
                "advanced_settings_guides",
                "lumetri_color_workflows",
            ],
            "forbidden": [
                "general_video_editing_resources",
                "other_editing_software_resources",
            ],
            "required": ["reference_premiere_pro", "cover_advanced_settings"],
            "actual_next": "adobe_premiere_pro_resources",
            "status": "satisfied",
        },
    },
    "06878be2": {
        "weak_cues": ["Sony A7R IV", "Godox V1", "flash accessories"],
        "gold_entities": ["Sony A7R IV", "Godox V1", "Sony HVL-F60RM"],
        "gold_constraints": {
            "allowed": [
                "sony_compatible_accessories",
                "high_quality_photography_gear",
                "flash_protection_or_power_accessories",
            ],
            "forbidden": ["other_brand_incompatible_gear", "low_quality_gear"],
            "required": ["compatible_with_sony_a7r_iv"],
            "actual_next": "sony_compatible_accessories",
            "status": "satisfied",
        },
    },
    "75832dbd": {
        "weak_cues": ["AI in healthcare", "medical image analysis", "deep learning"],
        "gold_entities": ["artificial intelligence in healthcare", "medical image analysis"],
        "gold_constraints": {
            "allowed": [
                "ai_healthcare_publications",
                "medical_image_analysis_conferences",
                "deep_learning_healthcare_papers",
            ],
            "forbidden": ["general_ai_topics", "non_healthcare_ai_topics"],
            "required": ["focus_on_healthcare", "recent_publications_or_conferences"],
            "actual_next": "ai_healthcare_publications",
            "status": "satisfied",
        },
    },
    "0edc2aef": {
        "weak_cues": ["Miami trip", "ocean or skyline view", "rooftop pool"],
        "gold_entities": ["Miami", "rooftop pool", "hot tub balcony"],
        "gold_constraints": {
            "allowed": [
                "miami_hotels_with_views",
                "rooftop_pool_hotels",
                "balcony_hot_tub_hotels",
            ],
            "forbidden": ["basic_budget_hotels", "hotels_without_views"],
            "required": ["in_miami", "great_views_or_unique_feature"],
            "actual_next": "miami_hotels_with_views",
            "status": "satisfied",
        },
    },
    "1d4e3b97": {
        "weak_cues": ["Sunday group rides", "chain and cassette replacement", "Garmin bike computer"],
        "gold_entities": ["Garmin bike computer", "chain", "cassette"],
        "gold_causal_chain": [
            "the bike's chain and cassette were replaced on February 1st",
            "the rider also started using a new Garmin bike computer to track performance",
            "the improved drivetrain and better ride monitoring explain the stronger Sunday group ride performance",
        ],
        "gold_scene": {"tone": "analytic", "mode": "causal_reflection", "uncertainty": 0.2},
    },
}


DERIVED_ITEMS: List[Dict[str, object]] = [
    {
        "item_id": "pm_derived_commute_audiobooks_cause",
        "source_question_id": "118b2229",
        "task_type": "causal_recall",
        "question": "Why were audiobooks a good fit for my daily routine?",
        "weak_cues": ["daily commute", "Audible", "45 minutes each way"],
        "gold_entities": ["audiobooks", "Audible", "commute"],
        "gold_constraints": {"allowed": [], "forbidden": [], "required": [], "actual_next": None, "status": None},
        "gold_causal_chain": [
            "the user listens to audiobooks during a daily commute",
            "the commute takes 45 minutes each way, creating a long listening window",
            "that commute pattern makes audiobooks a natural fit for the user's routine",
        ],
        "gold_turning_point": None,
        "gold_scene": {"tone": "analytic", "mode": "causal_reflection", "uncertainty": 0.2},
    },
    {
        "item_id": "pm_derived_mrs_johnson_surgery_cause",
        "source_question_id": "66f24dbb",
        "task_type": "causal_recall",
        "question": "Why did I bring groceries and flowers to Mrs. Johnson?",
        "weak_cues": ["Mrs. Johnson", "groceries and flowers", "recovery"],
        "gold_entities": ["Mrs. Johnson", "groceries", "flowers"],
        "gold_constraints": {"allowed": [], "forbidden": [], "required": [], "actual_next": None, "status": None},
        "gold_causal_chain": [
            "Mrs. Johnson was recovering from surgery",
            "the user wanted to support her during that recovery",
            "that is why the user brought groceries and flowers",
        ],
        "gold_turning_point": None,
        "gold_scene": {"tone": "analytic", "mode": "causal_reflection", "uncertainty": 0.15},
    },
    {
        "item_id": "pm_derived_cartwheel_preferences_constraint",
        "source_question_id": "51a45a95",
        "task_type": "constraint_recall",
        "question": "What Cartwheel app changes would best match my stated coupon-management preferences?",
        "weak_cues": ["expiration date sorting", "notification overload", "Target Cartwheel"],
        "gold_entities": ["Cartwheel", "Target", "expiration date"],
        "gold_constraints": {
            "allowed": [
                "sort_offers_by_expiration_date",
                "customize_notification_types",
                "expiring_soon_notifications",
                "category_specific_notifications",
            ],
            "forbidden": ["too_many_generic_notifications"],
            "required": ["help_prioritize_expiring_deals", "reduce_notification_overload"],
            "actual_next": "sort_offers_by_expiration_date",
            "status": "satisfied",
        },
        "gold_causal_chain": [],
        "gold_turning_point": None,
        "gold_scene": {"tone": "practical", "mode": "practical_planning", "uncertainty": 0.2},
    },
    {
        "item_id": "pm_derived_budget_boundaries_constraint",
        "source_question_id": "7527f7e2",
        "task_type": "constraint_recall",
        "question": "What shopping changes fit the budget boundaries I wanted to set?",
        "weak_cues": ["Saks and Neiman Marcus", "affordable alternatives", "beauty box subscription"],
        "gold_entities": ["Mint", "H&M", "Uniqlo", "discount store"],
        "gold_constraints": {
            "allowed": [
                "buy_affordable_alternatives",
                "shop_at_hm_or_uniqlo",
                "cancel_monthly_beauty_box",
                "buy_discount_beauty_products",
            ],
            "forbidden": ["frequent_high_end_department_store_spending", "unused_subscription_spending"],
            "required": ["lower_cost_with_similar_quality", "support_budget_control"],
            "actual_next": "buy_affordable_alternatives",
            "status": "satisfied",
        },
        "gold_causal_chain": [],
        "gold_turning_point": None,
        "gold_scene": {"tone": "practical", "mode": "practical_planning", "uncertainty": 0.2},
    },
    {
        "item_id": "pm_derived_spirituality_turn",
        "source_question_id": "dccbc061",
        "task_type": "turning_point_recall",
        "question": "What was the turning point in my stance on spirituality?",
        "weak_cues": ["Buddhism", "synchronicity", "shift from atheist"],
        "gold_entities": ["Buddhism", "synchronicity", "atheist"],
        "gold_constraints": {"allowed": [], "forbidden": [], "required": [], "actual_next": None, "status": None},
        "gold_causal_chain": [],
        "gold_turning_point": {
            "session_id": "answer_8f276838",
            "start_turn": 0,
            "end_turn": 0,
            "pre_state": "staunch_atheist",
            "post_state": "exploring_spirituality_through_buddhism_and_synchronicity",
        },
        "gold_scene": {"tone": "analytic", "mode": "causal_reflection", "uncertainty": 0.15},
    },
    {
        "item_id": "pm_derived_instagram_limit_turn",
        "source_question_id": "545bd2b5",
        "task_type": "turning_point_recall",
        "question": "What behavior change was I trying to make about my Instagram use?",
        "weak_cues": ["2 hours per day", "30-minute limit", "doodling during breaks"],
        "gold_entities": ["Instagram", "30 minutes per day", "doodling"],
        "gold_constraints": {"allowed": [], "forbidden": [], "required": [], "actual_next": None, "status": None},
        "gold_causal_chain": [],
        "gold_turning_point": {
            "session_id": "answer_47ffab4c",
            "start_turn": 4,
            "end_turn": 10,
            "pre_state": "spending_about_two_hours_per_day_on_instagram",
            "post_state": "trying_to_limit_instagram_to_thirty_minutes_and_replace_breaks_with_doodling",
        },
        "gold_scene": {"tone": "practical", "mode": "practical_planning", "uncertainty": 0.2},
    },
]


def _normalize_label(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return lowered or "unknown"


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.strip().lower()).strip()


def _detect_task_type(question: str, question_type: str = "") -> str:
    lowered = question.lower()
    qtype = question_type.strip().lower()
    if any(token in lowered for token in ("why", "reason", "because", "caused")):
        return "causal_recall"
    if any(token in lowered for token in ("when", "first time", "changed", "switch", "turning point")):
        return "turning_point_recall"
    if qtype == "single-session-preference":
        return "constraint_recall"
    if any(token in lowered for token in ("allow", "allowed", "forbidden", "blocked", "next step", "preference")):
        return "constraint_recall"
    if any(
        token in lowered
        for token in (
            "remember",
            "which one",
            "what item",
            "what did i",
            "where did i",
            "what degree",
            "what play",
            "what is the name",
            "which social media platform",
        )
    ):
        return "cue_recall"
    return "event_recall"


def _weak_cues(question: str, answer: str, dialect: AAAKLiteDialect) -> List[str]:
    topics = dialect._extract_topics(question + " " + answer, max_topics=3)
    entities = dialect._detect_entities(answer)
    cues = list(dict.fromkeys(topics + entities + [answer[:48].strip()] if answer else topics + entities))
    return [cue for cue in cues if cue][:3]


def _default_session_span(session: Sequence[Dict[str, str]], session_id: str) -> Dict[str, object]:
    user_turn_indices = [idx for idx, turn in enumerate(session) if turn.get("role") == "user"]
    if not user_turn_indices:
        return {"session_id": session_id, "start_turn": 0, "end_turn": 0}
    return {
        "session_id": session_id,
        "start_turn": int(user_turn_indices[0]),
        "end_turn": int(user_turn_indices[-1]),
    }


def _evidence_terms(question: str, answer: str, weak_cues: Sequence[str], entities: Sequence[str]) -> List[str]:
    candidates = list(weak_cues) + list(entities)
    if answer and len(answer.strip()) <= 64:
        candidates.append(answer)
    question_norm = _normalize_text(question)
    if question_norm:
        question_tokens = [token for token in question_norm.split() if len(token) >= 5]
        candidates.extend(question_tokens[:4])
    out: List[str] = []
    for raw in candidates:
        normalized = _normalize_text(str(raw))
        if len(normalized) < 3:
            continue
        out.append(normalized)
    return list(dict.fromkeys(sorted(out, key=len, reverse=True)))


def _tight_session_span(
    session: Sequence[Mapping[str, str]],
    session_id: str,
    evidence_terms: Sequence[str],
) -> Dict[str, object]:
    matched_indices: List[int] = []
    for idx, turn in enumerate(session):
        if turn.get("role") != "user":
            continue
        turn_text = _normalize_text(str(turn.get("content", "")))
        if any(term in turn_text for term in evidence_terms):
            matched_indices.append(idx)
    if matched_indices:
        return {
            "session_id": session_id,
            "start_turn": int(min(matched_indices)),
            "end_turn": int(max(matched_indices)),
        }
    return _default_session_span(session, session_id)


def _scene_from_text(text: str) -> Dict[str, object]:
    lowered = text.lower()
    if any(token in lowered for token in ("need", "schedule", "track", "list", "record")):
        mode = "practical_planning"
        tone = "practical"
        uncertainty = 0.25
    elif any(token in lowered for token in ("why", "because", "reason", "caused")):
        mode = "causal_reflection"
        tone = "analytic"
        uncertainty = 0.3
    elif any(token in lowered for token in ("wow", "amazing", "interesting", "excited")):
        mode = "enthusiastic_discussion"
        tone = "enthusiastic"
        uncertainty = 0.2
    else:
        mode = "general_recall"
        tone = "neutral"
        uncertainty = 0.4
    return {"tone": tone, "mode": mode, "uncertainty": uncertainty}


def _apply_curated_override(question_id: str, item: Dict[str, object]) -> Dict[str, object]:
    override = CURATED_ITEM_OVERRIDES.get(question_id)
    if not override:
        return item
    item.update(override)
    return item


def _empty_constraint_object() -> Dict[str, object]:
    return {
        "allowed": [],
        "forbidden": [],
        "required": [],
        "actual_next": None,
        "status": None,
    }


def _clone_json(payload: object) -> object:
    return json.loads(json.dumps(payload))


def _swap_top_sessions(session_ids: Sequence[str]) -> List[str]:
    session_ids = [str(session_id) for session_id in session_ids]
    if len(session_ids) < 2:
        return session_ids
    return [session_ids[1], session_ids[0], *session_ids[2:]]


def _base_prediction_payload(item: Mapping[str, object], ranked_sessions: Sequence[str], idx: int) -> Dict[str, object]:
    return {
        "pred_sessions": [str(session_id) for session_id in ranked_sessions],
        "pred_spans": _clone_json(item.get("gold_spans", [])),
        "pred_entities": list(item.get("gold_entities", [])),
        "pred_constraints": _clone_json(item.get("gold_constraints", _empty_constraint_object())),
        "pred_causal_chain": list(item.get("gold_causal_chain", [])),
        "pred_turning_point": _clone_json(item.get("gold_turning_point")),
        "pred_scene": _clone_json(item.get("gold_scene", {})),
        "memory_stats": {
            "bytes_used": 1024.0 + 32.0 * idx,
            "latency_ms": 40.0 + idx,
            "archive_support_ratio": 0.03 + 0.001 * (idx % 5),
        },
    }


def _build_protocol_runs(base_prediction: Mapping[str, object], ranked_sessions: Sequence[str]) -> Dict[str, object]:
    fixed_budget = _clone_json(base_prediction)

    weak_cue = _clone_json(base_prediction)
    weak_cue["pred_entities"] = list(weak_cue.get("pred_entities", []))[:2]
    weak_cue["pred_causal_chain"] = list(weak_cue.get("pred_causal_chain", []))[:2]
    weak_cue["memory_stats"]["latency_ms"] = float(weak_cue["memory_stats"].get("latency_ms", 0.0)) * 0.9

    refined_cue = _clone_json(base_prediction)
    refined_cue["pred_entities"] = list(refined_cue.get("pred_entities", []))[:3]
    refined_cue["memory_stats"]["latency_ms"] = float(refined_cue["memory_stats"].get("latency_ms", 0.0)) * 0.95

    chunk_shuffle = _clone_json(base_prediction)
    chunk_shuffle["pred_sessions"] = _swap_top_sessions(ranked_sessions)
    chunk_shuffle["pred_spans"] = []
    chunk_shuffle["pred_entities"] = list(chunk_shuffle.get("pred_entities", []))[:1]

    relation_shuffle = _clone_json(base_prediction)
    relation_shuffle["pred_sessions"] = _swap_top_sessions(ranked_sessions)
    relation_shuffle["pred_causal_chain"] = []
    relation_shuffle["pred_turning_point"] = None

    constraint_perturbation = _clone_json(base_prediction)
    constraint_perturbation["pred_constraints"] = {
        "allowed": [],
        "forbidden": list(base_prediction.get("pred_constraints", {}).get("forbidden", [])),
        "required": [],
        "actual_next": None,
        "status": "outside",
    }

    return {
        "fixed_budget": fixed_budget,
        "weak_cue": weak_cue,
        "progressive_recall": {
            "stages": [
                {"stage": "weak_cue", **weak_cue},
                {"stage": "refined_cue", **refined_cue},
                {"stage": "full_question", **_clone_json(base_prediction)},
            ]
        },
        "structural_perturbation": {
            "chunk_shuffle": chunk_shuffle,
            "relation_shuffle": relation_shuffle,
            "constraint_perturbation": constraint_perturbation,
        },
    }


def _build_item_from_entry(
    entry: dict,
    idx: int,
    dialect: AAAKLiteDialect,
    item_id: str | None = None,
    task_type: str | None = None,
    question: str | None = None,
    weak_cues: List[str] | None = None,
    gold_entities: List[str] | None = None,
    gold_constraints: Dict[str, object] | None = None,
    gold_causal_chain: List[str] | None = None,
    gold_turning_point: Dict[str, object] | None = None,
    gold_scene: Dict[str, object] | None = None,
) -> Tuple[dict, dict]:
    answer_session_ids = list(entry.get("answer_session_ids", []))
    haystack_session_ids = list(entry.get("haystack_session_ids", []))
    haystack_sessions = list(entry.get("haystack_sessions", []))
    answer_text = str(entry.get("answer", "")).strip()
    base_question = str(entry.get("question", "")).strip()
    question_id = str(entry.get("question_id", f"item_{idx:04d}"))
    answer_entities = dialect._detect_entities(answer_text)
    answer_topics = dialect._extract_topics(answer_text, max_topics=3)
    resolved_task_type = task_type or _detect_task_type(base_question, str(entry.get("question_type", "")))
    scene = gold_scene or _scene_from_text((question or base_question) + " " + answer_text)
    item = {
        "item_id": item_id or f"pm_{idx:04d}_{question_id}",
        "task_type": resolved_task_type,
        "question": question or base_question,
        "weak_cues": weak_cues or _weak_cues(base_question, answer_text, dialect),
        "gold_sessions": answer_session_ids,
        "gold_spans": [],
        "gold_entities": gold_entities or list(dict.fromkeys(answer_entities + answer_topics))[:5],
        "gold_constraints": gold_constraints
        or (
            {
                "allowed": [_normalize_label(answer_text)] if resolved_task_type == "constraint_recall" and answer_text else [],
                "forbidden": [],
                "required": [],
                "actual_next": _normalize_label(answer_text) if resolved_task_type == "constraint_recall" and answer_text else None,
                "status": "satisfied" if resolved_task_type == "constraint_recall" and answer_text else None,
            }
            if resolved_task_type == "constraint_recall"
            else _empty_constraint_object()
        ),
        "gold_causal_chain": gold_causal_chain or ([answer_text] if resolved_task_type == "causal_recall" and answer_text else []),
        "gold_turning_point": gold_turning_point
        or (
            {
                "session_id": answer_session_ids[0],
                "start_turn": 0,
                "end_turn": 0,
                "pre_state": None,
                "post_state": _normalize_label(answer_text) if answer_text else None,
            }
            if resolved_task_type == "turning_point_recall" and answer_session_ids
            else None
        ),
        "gold_scene": scene,
    }
    item = _apply_curated_override(question_id, item)
    if item.get("gold_turning_point"):
        turning_point = dict(item["gold_turning_point"])
        item["gold_spans"] = [
            {
                "session_id": turning_point["session_id"],
                "start_turn": int(turning_point["start_turn"]),
                "end_turn": int(turning_point["end_turn"]),
            }
        ]
    else:
        evidence_terms = _evidence_terms(
            str(item.get("question", "")),
            answer_text,
            item.get("weak_cues", []),
            item.get("gold_entities", []),
        )
        gold_spans = []
        for session_id, session in zip(haystack_session_ids, haystack_sessions):
            if session_id in answer_session_ids:
                gold_spans.append(_tight_session_span(session, session_id, evidence_terms))
        item["gold_spans"] = gold_spans or [
            _default_session_span(session, session_id)
            for session_id, session in zip(haystack_session_ids, haystack_sessions)
            if session_id in answer_session_ids
        ]

    ranked_sessions = answer_session_ids + [sid for sid in haystack_session_ids if sid not in answer_session_ids][:4]
    base_prediction = _base_prediction_payload(item, ranked_sessions, idx)
    source_record = {
        "item_id": item["item_id"],
        "question": item["question"],
        "ranked_sessions": ranked_sessions,
        **base_prediction,
        "protocol_runs": _build_protocol_runs(base_prediction, ranked_sessions),
    }
    return item, source_record


def build_items(data: Sequence[dict], limit: int) -> Tuple[List[dict], List[dict]]:
    dialect = AAAKLiteDialect()
    items: List[dict] = []
    source_records: List[dict] = []
    selected_entries: List[dict] = []
    data_by_question_id = {str(entry.get("question_id", "")): entry for entry in data}
    quotas = {
        "causal_recall": max(2, round(limit * 0.15)),
        "turning_point_recall": max(3, round(limit * 0.25)),
        "constraint_recall": max(2, round(limit * 0.20)),
        "cue_recall": max(2, round(limit * 0.20)),
    }
    quotas["event_recall"] = max(1, limit - sum(quotas.values()))
    picked = {key: 0 for key in quotas}

    for idx, derived in enumerate(DERIVED_ITEMS):
        if idx >= limit:
            break
        source_question_id = str(derived["source_question_id"])
        entry = data_by_question_id[source_question_id]
        item, source_record = _build_item_from_entry(
            entry,
            idx=idx,
            dialect=dialect,
            item_id=str(derived["item_id"]),
            task_type=str(derived["task_type"]),
            question=str(derived["question"]),
            weak_cues=list(derived.get("weak_cues", [])),
            gold_entities=list(derived.get("gold_entities", [])),
            gold_constraints=dict(derived.get("gold_constraints", {})),
            gold_causal_chain=list(derived.get("gold_causal_chain", [])),
            gold_turning_point=derived.get("gold_turning_point"),
            gold_scene=dict(derived.get("gold_scene", {})),
        )
        items.append(item)
        source_records.append(source_record)
        picked[item["task_type"]] = picked.get(item["task_type"], 0) + 1

    excluded_question_ids = {str(derived["source_question_id"]) for derived in DERIVED_ITEMS}

    for entry in data:
        question_id = str(entry.get("question_id", ""))
        if question_id in excluded_question_ids:
            continue
        task_type = _detect_task_type(
            str(entry.get("question", "")).strip(),
            str(entry.get("question_type", "")),
        )
        if picked.get(task_type, 0) >= quotas.get(task_type, 0):
            continue
        selected_entries.append(entry)
        picked[task_type] += 1
        if len(items) + len(selected_entries) >= limit:
            break

    if len(selected_entries) < limit:
        seen_ids = {str(entry.get("question_id", "")) for entry in selected_entries}
        for entry in data:
            question_id = str(entry.get("question_id", ""))
            if question_id in seen_ids or question_id in excluded_question_ids:
                continue
            selected_entries.append(entry)
            seen_ids.add(question_id)
            if len(items) + len(selected_entries) >= limit:
                break

    start_idx = len(items)
    remaining = max(0, limit - start_idx)
    for offset, entry in enumerate(selected_entries[:remaining]):
        item, source_record = _build_item_from_entry(
            entry,
            idx=start_idx + offset,
            dialect=dialect,
        )
        items.append(item)
        source_records.append(source_record)
    return items, source_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small PMS-Bench sample set from LongMemEval.")
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval_s_cleaned.json"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--items-out", type=Path, default=Path("pms_bench_sample_items.json"))
    parser.add_argument("--source-out", type=Path, default=Path("pms_bench_sample_source_records.json"))
    args = parser.parse_args()

    data = load_longmemeval(args.data, limit_questions=None)
    items, source_records = build_items(data, limit=args.limit)
    items_document = {
        "benchmark_name": BENCHMARK_NAME,
        "format_version": FORMAT_VERSION,
        "items": items,
    }
    source_document = {
        "benchmark_name": BENCHMARK_NAME,
        "format_version": FORMAT_VERSION,
        "records": source_records,
    }
    args.items_out.write_text(json.dumps(items_document, indent=2), encoding="utf-8")
    args.source_out.write_text(json.dumps(source_document, indent=2), encoding="utf-8")

    print("PMS Sample Dataset Builder")
    print(f"items: {len(items)}")
    print(f"records: {len(source_records)}")
    print(f"[done] wrote {args.items_out}")
    print(f"[done] wrote {args.source_out}")


if __name__ == "__main__":
    main()
