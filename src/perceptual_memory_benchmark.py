from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Optional, Sequence

BENCHMARK_NAME = "PMS-Bench"
FORMAT_VERSION = "2.0"


@dataclass
class Span:
    session_id: str
    start_turn: int
    end_turn: int
    score: float = 0.0


@dataclass
class TurningPoint:
    session_id: str
    start_turn: int
    end_turn: int
    pre_state: Optional[str] = None
    post_state: Optional[str] = None


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_items(payload) -> List[dict]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping) and isinstance(payload.get("items"), list):
        return [dict(item) for item in payload["items"] if isinstance(item, Mapping)]
    raise ValueError("Unsupported benchmark items payload format.")


def _normalize_prediction_index(payload) -> Dict[str, dict]:
    if isinstance(payload, list):
        return {str(item["item_id"]): dict(item) for item in payload if isinstance(item, dict) and "item_id" in item}
    if isinstance(payload, Mapping):
        if isinstance(payload.get("predictions"), list):
            return _normalize_prediction_index(payload["predictions"])
        if all(isinstance(value, Mapping) for value in payload.values()):
            normalized: Dict[str, dict] = {}
            for key, value in payload.items():
                row = dict(value)
                row.setdefault("item_id", key)
                normalized[str(row["item_id"])] = row
            return normalized
    raise ValueError("Unsupported predictions payload format.")


def _to_spans(items: Sequence[Mapping[str, object]]) -> List[Span]:
    spans: List[Span] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        session_id = str(item.get("session_id", "")).strip()
        if not session_id:
            continue
        spans.append(
            Span(
                session_id=session_id,
                start_turn=int(item.get("start_turn", 0)),
                end_turn=int(item.get("end_turn", 0)),
                score=float(item.get("score", 0.0) or 0.0),
            )
        )
    return spans


def _set_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    gold_set = {str(item).strip().lower() for item in gold if str(item).strip()}
    pred_set = {str(item).strip().lower() for item in pred if str(item).strip()}
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    overlap = len(gold_set & pred_set)
    precision = overlap / max(len(pred_set), 1)
    recall = overlap / max(len(gold_set), 1)
    if precision + recall <= 1e-12:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _normalize_text(text: object) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _text_token_f1(gold: object, pred: object) -> float:
    gold_tokens = _normalize_text(gold).split()
    pred_tokens = _normalize_text(pred).split()
    return _set_f1(gold_tokens, pred_tokens)


def _soft_step_set_f1(gold_steps: Sequence[str], pred_steps: Sequence[str]) -> float:
    if not gold_steps and not pred_steps:
        return 1.0
    if not gold_steps or not pred_steps:
        return 0.0
    recall = mean(max(_text_token_f1(gold_step, pred_step) for pred_step in pred_steps) for gold_step in gold_steps)
    precision = mean(max(_text_token_f1(pred_step, gold_step) for gold_step in gold_steps) for pred_step in pred_steps)
    if precision + recall <= 1e-12:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _causal_order_score(gold_steps: Sequence[str], pred_steps: Sequence[str]) -> float:
    if not gold_steps or not pred_steps:
        return 0.0
    matched_indices: List[int] = []
    for gold_step in gold_steps:
        best_idx = -1
        best_score = 0.0
        for idx, pred_step in enumerate(pred_steps):
            score = _text_token_f1(gold_step, pred_step)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= 0.5:
            matched_indices.append(best_idx)
    if not matched_indices:
        return 0.0
    ordered = sum(1 for prev, curr in zip(matched_indices, matched_indices[1:]) if curr >= prev)
    return (ordered + 1) / max(len(gold_steps), 1)


def _span_iou(a: Span, b: Span) -> float:
    if a.session_id != b.session_id:
        return 0.0
    inter = max(0, min(a.end_turn, b.end_turn) - max(a.start_turn, b.start_turn) + 1)
    if inter <= 0:
        return 0.0
    union = max(a.end_turn, b.end_turn) - min(a.start_turn, b.start_turn) + 1
    return inter / max(union, 1)


def _best_span_iou(gold_spans: Sequence[Span], pred_spans: Sequence[Span]) -> float:
    if not gold_spans or not pred_spans:
        return 0.0
    return max(_span_iou(gold, pred) for gold in gold_spans for pred in pred_spans)


def _to_turning_point(payload: object) -> Optional[TurningPoint]:
    if not isinstance(payload, Mapping):
        return None
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        return None
    return TurningPoint(
        session_id=session_id,
        start_turn=int(payload.get("start_turn", 0)),
        end_turn=int(payload.get("end_turn", 0)),
        pre_state=str(payload.get("pre_state")).strip() if payload.get("pre_state") is not None else None,
        post_state=str(payload.get("post_state")).strip() if payload.get("post_state") is not None else None,
    )


def _turning_point_metrics(gold_turning: object, pred_turning: object, pred_spans: Sequence[Span]) -> Dict[str, float]:
    gold = _to_turning_point(gold_turning)
    pred = _to_turning_point(pred_turning)
    if gold is None:
        return {
            "turning_span_iou": 0.0,
            "turning_pre_state_f1": 0.0,
            "turning_post_state_f1": 0.0,
            "turning_state_f1": 0.0,
            "turning_point_hit": 0.0,
        }

    gold_span = Span(session_id=gold.session_id, start_turn=gold.start_turn, end_turn=gold.end_turn)
    pred_span_iou = 0.0
    if pred is not None:
        pred_span_iou = _span_iou(gold_span, Span(pred.session_id, pred.start_turn, pred.end_turn))
    fallback_span_iou = _best_span_iou([gold_span], pred_spans)
    turning_span_iou = max(pred_span_iou, fallback_span_iou)

    pre_state_f1 = _text_token_f1(gold.pre_state, pred.pre_state) if gold.pre_state is not None and pred is not None else 0.0
    post_state_f1 = _text_token_f1(gold.post_state, pred.post_state) if gold.post_state is not None and pred is not None else 0.0
    state_parts = []
    if gold.pre_state is not None:
        state_parts.append(pre_state_f1)
    if gold.post_state is not None:
        state_parts.append(post_state_f1)
    turning_state_f1 = float(mean(state_parts)) if state_parts else 0.0
    session_match = float(pred is not None and pred.session_id == gold.session_id)
    turning_point_hit = float(mean([session_match, turning_span_iou, turning_state_f1]))
    return {
        "turning_span_iou": float(turning_span_iou),
        "turning_pre_state_f1": float(pre_state_f1),
        "turning_post_state_f1": float(post_state_f1),
        "turning_state_f1": float(turning_state_f1),
        "turning_point_hit": float(turning_point_hit),
    }


def _recall_at_k(gold_sessions: Sequence[str], pred_sessions: Sequence[str], k: int) -> float:
    gold_set = {str(item) for item in gold_sessions}
    top_k = {str(item) for item in pred_sessions[:k]}
    if not gold_set:
        return 0.0
    return 1.0 if gold_set & top_k else 0.0


def _mrr(gold_sessions: Sequence[str], pred_sessions: Sequence[str]) -> float:
    gold_set = {str(item) for item in gold_sessions}
    for rank, session_id in enumerate(pred_sessions, start=1):
        if str(session_id) in gold_set:
            return 1.0 / rank
    return 0.0


def _constraint_metrics(gold_constraints: Mapping[str, object], pred_constraints: Mapping[str, object]) -> Dict[str, float]:
    gold_constraints = gold_constraints or {}
    pred_constraints = pred_constraints or {}
    return {
        "allowed_set_f1": _set_f1(gold_constraints.get("allowed", []), pred_constraints.get("allowed", [])),
        "forbidden_set_f1": _set_f1(gold_constraints.get("forbidden", []), pred_constraints.get("forbidden", [])),
        "required_set_f1": _set_f1(gold_constraints.get("required", []), pred_constraints.get("required", [])),
        "actual_next_accuracy": float(
            str(gold_constraints.get("actual_next", "")).strip().lower()
            == str(pred_constraints.get("actual_next", "")).strip().lower()
        )
        if gold_constraints.get("actual_next") is not None
        else 0.0,
        "constraint_status_accuracy": float(
            str(gold_constraints.get("status", "")).strip().lower()
            == str(pred_constraints.get("status", "")).strip().lower()
        )
        if gold_constraints.get("status") is not None
        else 0.0,
    }


def _causal_metrics(gold_chain: Sequence[str], pred_chain: Sequence[str], event_hit: float) -> Dict[str, float]:
    causal_step_f1 = _soft_step_set_f1(gold_chain, pred_chain) if gold_chain or pred_chain else 0.0
    causal_order_score = _causal_order_score(gold_chain, pred_chain) if gold_chain or pred_chain else 0.0
    if gold_chain:
        cause_hit = float(mean([event_hit, causal_step_f1, causal_order_score]))
    else:
        cause_hit = 0.0
    return {
        "causal_step_f1": float(causal_step_f1),
        "causal_order_score": float(causal_order_score),
        "cause_hit": float(cause_hit),
    }


def _scene_metrics(gold_scene: Mapping[str, object], pred_scene: Mapping[str, object]) -> Dict[str, float]:
    gold_scene = gold_scene or {}
    pred_scene = pred_scene or {}
    gold_unc = gold_scene.get("uncertainty")
    pred_unc = pred_scene.get("uncertainty")
    unc_mae = 0.0
    if gold_unc is not None and pred_unc is not None:
        unc_mae = abs(float(gold_unc) - float(pred_unc))
    tone_accuracy = float(
        str(gold_scene.get("tone", "")).strip().lower() == str(pred_scene.get("tone", "")).strip().lower()
    ) if gold_scene.get("tone") is not None else 0.0
    scene_mode_accuracy = float(
        str(gold_scene.get("mode", "")).strip().lower() == str(pred_scene.get("mode", "")).strip().lower()
    ) if gold_scene.get("mode") is not None else 0.0
    scene_fidelity = mean([tone_accuracy, scene_mode_accuracy, max(0.0, 1.0 - unc_mae)])
    return {
        "tone_accuracy": tone_accuracy,
        "scene_mode_accuracy": scene_mode_accuracy,
        "uncertainty_mae": float(unc_mae),
        "scene_fidelity": float(scene_fidelity),
    }


def _compression_metrics(memory_stats: Mapping[str, object], event_hit: float) -> Dict[str, float]:
    memory_stats = memory_stats or {}
    raw_bytes = float(memory_stats.get("raw_bytes", 0.0) or 0.0)
    bytes_used = float(memory_stats.get("bytes_used", 0.0) or 0.0)
    latency_ms = float(memory_stats.get("latency_ms", 0.0) or 0.0)
    support_ratio = float(memory_stats.get("archive_support_ratio", 0.0) or 0.0)
    compression_ratio = float(memory_stats.get("compression_ratio", 0.0) or 0.0)
    if compression_ratio <= 0.0 and raw_bytes > 0.0 and bytes_used > 0.0:
        compression_ratio = raw_bytes / max(bytes_used, 1e-9)
    recovery_per_kb = event_hit / max(bytes_used / 1024.0, 1e-9) if bytes_used > 0 else 0.0
    return {
        "avg_raw_bytes": raw_bytes,
        "avg_bytes_used": bytes_used,
        "avg_compression_ratio": compression_ratio,
        "avg_latency_ms": latency_ms,
        "archive_support_ratio": support_ratio,
        "recovery_per_kb": recovery_per_kb,
    }


def _empty_prediction(item_id: str) -> Dict[str, object]:
    return {
        "item_id": item_id,
        "pred_sessions": [],
        "pred_spans": [],
        "pred_entities": [],
        "pred_constraints": {},
        "pred_causal_chain": [],
        "pred_scene": {},
        "pred_turning_point": None,
        "protocol_runs": {},
        "memory_stats": {},
    }


def _item_metrics(item: Mapping[str, object], prediction: Mapping[str, object]) -> Dict[str, float]:
    gold_sessions = list(item.get("gold_sessions", []))
    pred_sessions = list(prediction.get("pred_sessions", []))
    gold_spans = _to_spans(item.get("gold_spans", []))
    pred_spans = _to_spans(prediction.get("pred_spans", []))
    span_iou = _best_span_iou(gold_spans, pred_spans)
    event_hit = float(span_iou > 0.0 or _recall_at_k(gold_sessions, pred_sessions, 1) > 0.0)

    metrics = {
        "recall@1": _recall_at_k(gold_sessions, pred_sessions, 1),
        "recall@3": _recall_at_k(gold_sessions, pred_sessions, 3),
        "recall@5": _recall_at_k(gold_sessions, pred_sessions, 5),
        "mrr": _mrr(gold_sessions, pred_sessions),
        "event_hit": event_hit,
        "span_iou": span_iou,
        "entity_f1": _set_f1(item.get("gold_entities", []), prediction.get("pred_entities", [])),
    }
    metrics.update(_constraint_metrics(item.get("gold_constraints", {}), prediction.get("pred_constraints", {})))
    metrics.update(_causal_metrics(item.get("gold_causal_chain", []), prediction.get("pred_causal_chain", []), event_hit))
    metrics.update(
        _turning_point_metrics(item.get("gold_turning_point"), prediction.get("pred_turning_point"), pred_spans)
        if str(item.get("task_type", "")) == "turning_point_recall"
        else {
            "turning_span_iou": 0.0,
            "turning_pre_state_f1": 0.0,
            "turning_post_state_f1": 0.0,
            "turning_state_f1": 0.0,
            "turning_point_hit": 0.0,
        }
    )
    metrics.update(_scene_metrics(item.get("gold_scene", {}), prediction.get("pred_scene", {})))
    metrics.update(_compression_metrics(prediction.get("memory_stats", {}), event_hit))
    return metrics


def _aggregate_metric_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(mean(row[key] for row in rows)) for key in keys}


def _evaluate_prediction_map(
    items: Sequence[Mapping[str, object]],
    prediction_index: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    rows: List[Dict[str, float]] = []
    task_rows: Dict[str, List[Dict[str, float]]] = {}
    missing_predictions: List[str] = []
    for item in items:
        item_id = str(item.get("item_id", ""))
        task_type = str(item.get("task_type", ""))
        prediction = prediction_index.get(item_id)
        if prediction is None:
            missing_predictions.append(item_id)
            prediction = _empty_prediction(item_id)
        row = _item_metrics(item, prediction)
        rows.append(row)
        task_rows.setdefault(task_type, []).append(row)
    task_summaries = {
        task_type: {"item_count": len(task_rows[task_type]), "summary": _aggregate_metric_rows(task_rows[task_type])}
        for task_type in task_rows
    }
    return {
        "rows": rows,
        "summary": _aggregate_metric_rows(rows),
        "task_summaries": task_summaries,
        "missing_prediction_ids": missing_predictions,
    }


def _protocol_prediction_index(
    prediction_index: Mapping[str, Mapping[str, object]],
    protocol_name: str,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for item_id, prediction in prediction_index.items():
        protocol_runs = prediction.get("protocol_runs", {})
        if isinstance(protocol_runs, Mapping) and isinstance(protocol_runs.get(protocol_name), Mapping):
            merged = dict(prediction)
            merged.update(protocol_runs[protocol_name])
            out[item_id] = merged
    return out


def _progressive_stage_prediction_indices(
    prediction_index: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, dict]]:
    stage_indices: Dict[str, Dict[str, dict]] = {}
    for item_id, prediction in prediction_index.items():
        protocol_runs = prediction.get("protocol_runs", {})
        progressive = protocol_runs.get("progressive_recall", {}) if isinstance(protocol_runs, Mapping) else {}
        for stage in progressive.get("stages", []) if isinstance(progressive, Mapping) else []:
            if not isinstance(stage, Mapping):
                continue
            stage_name = str(stage.get("stage", "")).strip()
            if not stage_name:
                continue
            merged = dict(prediction)
            merged.update(stage)
            stage_indices.setdefault(stage_name, {})[item_id] = merged
    return stage_indices


def _structural_variant_prediction_indices(
    prediction_index: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, dict]]:
    variant_indices: Dict[str, Dict[str, dict]] = {}
    for item_id, prediction in prediction_index.items():
        protocol_runs = prediction.get("protocol_runs", {})
        structural = protocol_runs.get("structural_perturbation", {}) if isinstance(protocol_runs, Mapping) else {}
        if not isinstance(structural, Mapping):
            continue
        for variant_name, variant_payload in structural.items():
            if not isinstance(variant_payload, Mapping):
                continue
            merged = dict(prediction)
            merged.update(variant_payload)
            variant_indices.setdefault(str(variant_name), {})[item_id] = merged
    return variant_indices


def _retrieval_score(summary: Mapping[str, float]) -> float:
    return float(
        mean(
            [
                summary.get("recall@1", 0.0),
                summary.get("recall@3", 0.0),
                summary.get("recall@5", 0.0),
                summary.get("mrr", 0.0),
            ]
        )
    )


def _constraint_score(summary: Mapping[str, float]) -> float:
    return float(
        mean(
            [
                summary.get("allowed_set_f1", 0.0),
                summary.get("forbidden_set_f1", 0.0),
                summary.get("required_set_f1", 0.0),
                summary.get("actual_next_accuracy", 0.0),
                summary.get("constraint_status_accuracy", 0.0),
            ]
        )
    )


def _summary_or_empty(task_summaries: Mapping[str, Mapping[str, object]], task_type: str) -> Mapping[str, float]:
    payload = task_summaries.get(task_type, {})
    return payload.get("summary", {}) if isinstance(payload, Mapping) else {}


def _normalized_subscores(
    summary: Mapping[str, float],
    task_summaries: Mapping[str, Mapping[str, object]],
    protocol_summaries: Mapping[str, Mapping[str, object]],
) -> Dict[str, float]:
    retrieval = _retrieval_score(summary)
    constraint = _constraint_score(_summary_or_empty(task_summaries, "constraint_recall"))
    perceptual = mean(
        [
            _summary_or_empty(task_summaries, "event_recall").get("event_hit", summary.get("event_hit", 0.0)),
            _summary_or_empty(task_summaries, "causal_recall").get("cause_hit", 0.0),
            _summary_or_empty(task_summaries, "turning_point_recall").get("turning_point_hit", 0.0),
            summary.get("span_iou", 0.0),
            summary.get("entity_f1", 0.0),
        ]
    )
    scene = mean(
        [
            summary.get("tone_accuracy", 0.0),
            summary.get("scene_mode_accuracy", 0.0),
            max(0.0, 1.0 - summary.get("uncertainty_mae", 1.0)),
        ]
    )
    compression = mean(
        [
            min(summary.get("recovery_per_kb", 0.0) / 4.0, 1.0),
            1.0 / (1.0 + summary.get("avg_latency_ms", 0.0) / 250.0),
            1.0 / (1.0 + summary.get("avg_bytes_used", 0.0) / 4096.0),
        ]
    )
    structural_summary = protocol_summaries.get("structural_perturbation", {})
    structural_deltas = structural_summary.get("deltas", {}) if isinstance(structural_summary, Mapping) else {}
    if structural_deltas:
        structural = mean(
            [
                max(0.0, 1.0 - structural_deltas.get("shuffle_degradation", 1.0)),
                structural_deltas.get("constraint_consistency", 0.0),
                max(0.0, 1.0 - structural_deltas.get("turning_state_drop", 1.0)),
            ]
        )
    else:
        structural = mean(
            [
                summary.get("archive_support_ratio", 0.0),
                summary.get("turning_state_f1", 0.0),
                summary.get("causal_order_score", 0.0),
            ]
        )
    return {
        "retrieval": float(retrieval),
        "constraint_fidelity": float(constraint),
        "perceptual_recovery": float(perceptual),
        "scene_fidelity": float(scene),
        "compression_utility": float(compression),
        "structural_stability": float(structural),
    }


def evaluate(items: Sequence[Mapping[str, object]], prediction_index: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    fixed_eval = _evaluate_prediction_map(items, prediction_index)
    summary = fixed_eval["summary"]
    task_summaries = fixed_eval["task_summaries"]
    protocol_summaries: Dict[str, object] = {
        "fixed_budget": {
            "summary": summary,
        }
    }

    weak_index = _protocol_prediction_index(prediction_index, "weak_cue")
    if weak_index:
        weak_eval = _evaluate_prediction_map(items, weak_index)
        protocol_summaries["weak_cue"] = {
            "summary": weak_eval["summary"],
            "deltas": {
                "cue_recall_drop": max(0.0, summary.get("recall@1", 0.0) - weak_eval["summary"].get("recall@1", 0.0)),
                "cue_span_drop": max(0.0, summary.get("span_iou", 0.0) - weak_eval["summary"].get("span_iou", 0.0)),
            },
        }

    stage_indices = _progressive_stage_prediction_indices(prediction_index)
    if stage_indices:
        stage_summaries = {
            stage_name: _evaluate_prediction_map(items, stage_prediction_index)["summary"]
            for stage_name, stage_prediction_index in stage_indices.items()
        }
        weak_summary = stage_summaries.get("weak_cue", {})
        full_summary = stage_summaries.get("full_question", stage_summaries.get("refined_cue", {}))
        protocol_summaries["progressive_recall"] = {
            "stage_summaries": stage_summaries,
            "deltas": {
                "progressive_recall_gain": max(0.0, full_summary.get("recall@1", 0.0) - weak_summary.get("recall@1", 0.0)),
                "progressive_span_gain": max(0.0, full_summary.get("span_iou", 0.0) - weak_summary.get("span_iou", 0.0)),
                "progressive_constraint_gain": max(
                    0.0,
                    _constraint_score(full_summary) - _constraint_score(weak_summary),
                ),
            },
        }

    variant_indices = _structural_variant_prediction_indices(prediction_index)
    if variant_indices:
        variant_summaries = {
            variant_name: _evaluate_prediction_map(items, variant_prediction_index)["summary"]
            for variant_name, variant_prediction_index in variant_indices.items()
        }
        avg_variant_retrieval = mean(_retrieval_score(variant_summary) for variant_summary in variant_summaries.values())
        avg_variant_constraint = mean(_constraint_score(variant_summary) for variant_summary in variant_summaries.values())
        avg_variant_turning = mean(variant_summary.get("turning_state_f1", 0.0) for variant_summary in variant_summaries.values())
        protocol_summaries["structural_perturbation"] = {
            "variant_summaries": variant_summaries,
            "deltas": {
                "shuffle_degradation": max(0.0, _retrieval_score(summary) - avg_variant_retrieval),
                "constraint_consistency": max(0.0, 1.0 - max(0.0, _constraint_score(summary) - avg_variant_constraint)),
                "turning_state_drop": max(0.0, summary.get("turning_state_f1", 0.0) - avg_variant_turning),
            },
        }

    subscores = _normalized_subscores(summary, task_summaries, protocol_summaries)
    pms_score = (
        0.22 * subscores["retrieval"]
        + 0.22 * subscores["constraint_fidelity"]
        + 0.22 * subscores["perceptual_recovery"]
        + 0.14 * subscores["scene_fidelity"]
        + 0.10 * subscores["compression_utility"]
        + 0.10 * subscores["structural_stability"]
    )
    return {
        "benchmark_name": BENCHMARK_NAME,
        "format_version": FORMAT_VERSION,
        "results": {
            "item_count": len(items),
            "missing_prediction_count": len(fixed_eval["missing_prediction_ids"]),
            "missing_prediction_ids": fixed_eval["missing_prediction_ids"],
            "summary": summary,
            "task_summaries": task_summaries,
            "protocol_summaries": protocol_summaries,
            "subscores": subscores,
            "pms_score": float(pms_score),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PMS-Bench predictions.")
    parser.add_argument("--items", type=Path, required=True, help="JSON file containing benchmark items.")
    parser.add_argument("--predictions", type=Path, required=True, help="JSON file containing system predictions.")
    parser.add_argument("--json-out", type=Path, default=Path("perceptual_memory_benchmark_results.json"))
    args = parser.parse_args()

    items_payload = _extract_items(_load_json(args.items))
    prediction_index = _normalize_prediction_index(_load_json(args.predictions))
    results_document = evaluate(items_payload, prediction_index)
    args.json_out.write_text(json.dumps(results_document, indent=2), encoding="utf-8")

    results = results_document["results"]
    print("Perceptual Memory Benchmark")
    print(f"items: {results['item_count']}")
    print(f"missing_predictions: {results['missing_prediction_count']}")
    for key, value in results["summary"].items():
        print(f"{key}: {value:.4f}")
    print("[subscores]")
    for key, value in results["subscores"].items():
        print(f"{key}: {value:.4f}")
    print(f"pms_score: {results['pms_score']:.4f}")
    print(f"[done] wrote {args.json_out}")


if __name__ == "__main__":
    main()
