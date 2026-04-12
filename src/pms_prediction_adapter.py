from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from benchmark_compare_k3m_aaak import AAAKLiteDialect, build_session_catalog, load_longmemeval

BENCHMARK_NAME = "PMS-Bench"
FORMAT_VERSION = "2.0"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_items(payload) -> List[dict]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping) and isinstance(payload.get("items"), list):
        return [dict(item) for item in payload["items"] if isinstance(item, Mapping)]
    raise ValueError("Unsupported items payload format.")


def _candidate_ids(record: Mapping[str, object]) -> List[str]:
    keys = (
        "item_id",
        "question_id",
        "id",
    )
    out: List[str] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def _candidate_questions(record: Mapping[str, object]) -> List[str]:
    keys = (
        "question",
        "query",
        "prompt",
    )
    out: List[str] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def _normalize_source_records(payload) -> List[dict]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return _normalize_source_records(payload["records"])
        if "predictions" in payload and isinstance(payload["predictions"], list):
            return _normalize_source_records(payload["predictions"])
        if all(isinstance(value, Mapping) for value in payload.values()):
            records: List[dict] = []
            for key, value in payload.items():
                row = dict(value)
                row.setdefault("item_id", key)
                records.append(row)
            return records
    raise ValueError("Unsupported source predictions payload format.")


def _extract_ranked_sessions(record: Mapping[str, object]) -> List[str]:
    keys = (
        "pred_sessions",
        "ranked_sessions",
        "ranked_session_ids",
        "session_ids",
        "retrieved_sessions",
    )
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    return []


def _extract_memory_stats(record: Mapping[str, object]) -> Dict[str, float]:
    stats = record.get("memory_stats")
    if isinstance(stats, Mapping):
        return {str(key): float(value) for key, value in stats.items() if isinstance(value, (int, float))}
    out: Dict[str, float] = {}
    for key in ("raw_bytes", "bytes_used", "compression_ratio", "latency_ms", "archive_support_ratio"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _best_source_record(
    item: Mapping[str, object],
    by_id: Mapping[str, dict],
    by_question: Mapping[str, dict],
) -> Optional[dict]:
    item_id = str(item.get("item_id", "")).strip()
    if item_id and item_id in by_id:
        return by_id[item_id]
    question = str(item.get("question", "")).strip()
    if question and question in by_question:
        return by_question[question]
    return None


def _session_texts_from_longmemeval(data_path: Optional[Path]) -> Dict[str, str]:
    if data_path is None:
        return {}
    data = load_longmemeval(data_path)
    sessions = build_session_catalog(data)
    return {session_id: record.text for session_id, record in sessions.items()}


def _session_turns_from_longmemeval(data_path: Optional[Path]) -> Dict[str, List[dict]]:
    if data_path is None:
        return {}
    out: Dict[str, List[dict]] = {}
    data = load_longmemeval(data_path, limit_questions=None)
    for entry in data:
        for session_id, session in zip(entry.get("haystack_session_ids", []), entry.get("haystack_sessions", [])):
            out.setdefault(str(session_id), list(session))
    return out


def _detect_entities(text: str) -> List[str]:
    dialect = AAAKLiteDialect()
    return dialect._detect_entities(text)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.strip().lower()).strip()


def _evidence_terms(item: Mapping[str, object], source: Mapping[str, object]) -> List[str]:
    candidates = list(item.get("weak_cues", [])) + list(item.get("gold_entities", []))
    candidates.extend(source.get("pred_entities", []))
    question = _normalize_text(str(item.get("question", "")))
    if question:
        candidates.extend(token for token in question.split() if len(token) >= 5)
    out: List[str] = []
    for candidate in candidates:
        normalized = _normalize_text(str(candidate))
        if len(normalized) >= 3:
            out.append(normalized)
    return list(dict.fromkeys(sorted(out, key=len, reverse=True)))


def _coarse_spans_from_session(
    session_id: str,
    session_turns: Sequence[Mapping[str, object]],
    evidence_terms: Sequence[str],
) -> List[dict]:
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
            "score": 0.5,
        }
    ]


def _fallback_causal_chain(
    item: Mapping[str, object],
    source: Mapping[str, object],
    top_session_turns: Sequence[Mapping[str, object]],
) -> List[str]:
    if str(item.get("task_type", "")) != "causal_recall":
        return []
    extracted: List[str] = []
    evidence_terms = _evidence_terms(item, source)
    for turn in top_session_turns:
        if str(turn.get("role", "")) != "user":
            continue
        content = str(turn.get("content", "")).strip()
        normalized = _normalize_text(content)
        if any(token in normalized for token in ("because", "caused", "reason", "why")) or any(
            term in normalized for term in evidence_terms
        ):
            extracted.append(content[:160].strip())
        if len(extracted) >= 3:
            break
    return extracted


def _normalize_protocol_prediction(payload: Mapping[str, object]) -> Dict[str, object]:
    return {
        "pred_sessions": [str(item) for item in payload.get("pred_sessions", []) if str(item).strip()],
        "pred_spans": payload.get("pred_spans", []) if isinstance(payload.get("pred_spans"), list) else [],
        "pred_entities": [str(item) for item in payload.get("pred_entities", []) if str(item).strip()],
        "pred_constraints": dict(payload.get("pred_constraints", {})) if isinstance(payload.get("pred_constraints"), Mapping) else {},
        "pred_causal_chain": [str(item) for item in payload.get("pred_causal_chain", []) if str(item).strip()],
        "pred_turning_point": dict(payload.get("pred_turning_point")) if isinstance(payload.get("pred_turning_point"), Mapping) else None,
        "pred_scene": dict(payload.get("pred_scene", {})) if isinstance(payload.get("pred_scene"), Mapping) else {},
        "memory_stats": _extract_memory_stats(payload),
    }


def _normalize_protocol_runs(payload: object) -> Dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    out: Dict[str, object] = {}
    for key in ("fixed_budget", "weak_cue"):
        if isinstance(payload.get(key), Mapping):
            out[key] = _normalize_protocol_prediction(payload[key])
    if isinstance(payload.get("progressive_recall"), Mapping):
        progressive = payload["progressive_recall"]
        stages = []
        for stage in progressive.get("stages", []):
            if isinstance(stage, Mapping):
                normalized = _normalize_protocol_prediction(stage)
                normalized["stage"] = str(stage.get("stage", "")).strip()
                stages.append(normalized)
        out["progressive_recall"] = {"stages": stages}
    if isinstance(payload.get("structural_perturbation"), Mapping):
        structural: Dict[str, object] = {}
        for key in ("chunk_shuffle", "relation_shuffle", "constraint_perturbation"):
            if isinstance(payload["structural_perturbation"].get(key), Mapping):
                structural[key] = _normalize_protocol_prediction(payload["structural_perturbation"][key])
        out["structural_perturbation"] = structural
    return out


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
    return {
        "tone": tone,
        "mode": mode,
        "uncertainty": uncertainty,
    }


def build_predictions(
    items: Sequence[Mapping[str, object]],
    source_records: Sequence[Mapping[str, object]],
    session_texts: Mapping[str, str],
    session_turns: Mapping[str, Sequence[Mapping[str, object]]],
) -> List[dict]:
    by_id: Dict[str, dict] = {}
    by_question: Dict[str, dict] = {}
    for record in source_records:
        for item_id in _candidate_ids(record):
            by_id[item_id] = dict(record)
        for question in _candidate_questions(record):
            by_question[question] = dict(record)

    predictions: List[dict] = []
    for item in items:
        source = _best_source_record(item, by_id, by_question) or {}
        pred_sessions = _extract_ranked_sessions(source)
        top_session = pred_sessions[0] if pred_sessions else ""
        top_text = session_texts.get(top_session, "")
        top_turns = list(session_turns.get(top_session, []))
        entities = _detect_entities(top_text) if top_text else []
        scene = _detect_scene(top_text) if top_text else {}
        pred_spans = source.get("pred_spans", []) if isinstance(source.get("pred_spans"), list) else []
        pred_causal_chain = [str(step) for step in source.get("pred_causal_chain", []) if str(step).strip()] if isinstance(source.get("pred_causal_chain"), list) else []

        if "pred_entities" in source and isinstance(source["pred_entities"], list):
            entities = [str(item) for item in source["pred_entities"]]
        if "pred_scene" in source and isinstance(source["pred_scene"], Mapping):
            scene = dict(source["pred_scene"])
        if not pred_spans and top_session and top_turns:
            pred_spans = _coarse_spans_from_session(top_session, top_turns, _evidence_terms(item, source))
        if not pred_causal_chain and top_turns:
            pred_causal_chain = _fallback_causal_chain(item, source, top_turns)

        protocol_runs = _normalize_protocol_runs(source.get("protocol_runs"))
        if "fixed_budget" not in protocol_runs:
            protocol_runs["fixed_budget"] = _normalize_protocol_prediction(
                {
                    "pred_sessions": pred_sessions,
                    "pred_spans": pred_spans,
                    "pred_entities": entities,
                    "pred_constraints": source.get("pred_constraints", {}),
                    "pred_causal_chain": pred_causal_chain,
                    "pred_turning_point": source.get("pred_turning_point"),
                    "pred_scene": scene,
                    "memory_stats": _extract_memory_stats(source),
                }
            )

        prediction = {
            "item_id": str(item.get("item_id", "")),
            "pred_sessions": pred_sessions,
            "pred_spans": pred_spans,
            "pred_entities": entities,
            "pred_constraints": source.get("pred_constraints", {}) if isinstance(source.get("pred_constraints"), Mapping) else {},
            "pred_causal_chain": pred_causal_chain,
            "pred_scene": scene,
            "pred_turning_point": source.get("pred_turning_point") if isinstance(source.get("pred_turning_point"), Mapping) else None,
            "protocol_runs": protocol_runs,
            "memory_stats": _extract_memory_stats(source),
        }
        predictions.append(prediction)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt existing system outputs into PMS benchmark_prediction format.")
    parser.add_argument("--items", type=Path, required=True, help="Benchmark items JSON list.")
    parser.add_argument("--source", type=Path, required=True, help="Source predictions or ranked-session records.")
    parser.add_argument("--longmemeval-data", type=Path, default=None, help="Optional LongMemEval data for session-text enrichment.")
    parser.add_argument("--json-out", type=Path, default=Path("pms_predictions.json"))
    args = parser.parse_args()

    items_payload = _extract_items(_load_json(args.items))
    source_records = _normalize_source_records(_load_json(args.source))
    session_texts = _session_texts_from_longmemeval(args.longmemeval_data)
    session_turns = _session_turns_from_longmemeval(args.longmemeval_data)
    predictions = build_predictions(items_payload, source_records, session_texts, session_turns)
    prediction_document = {
        "benchmark_name": BENCHMARK_NAME,
        "format_version": FORMAT_VERSION,
        "protocol_default": "fixed_budget",
        "predictions": predictions,
    }
    args.json_out.write_text(json.dumps(prediction_document, indent=2), encoding="utf-8")

    print("PMS Prediction Adapter")
    print(f"items: {len(items_payload)}")
    print(f"source_records: {len(source_records)}")
    print(f"predictions: {len(predictions)}")
    print(f"[done] wrote {args.json_out}")


if __name__ == "__main__":
    main()
