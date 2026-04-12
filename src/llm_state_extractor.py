from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import fcntl

from k3m_v2_balanced import sentence_chunks, tokenize


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


@dataclass
class LLMTransition:
    type: str
    reason: str


@dataclass
class LLMChunkState:
    chunk_id: int
    chunk_text: str
    summary: str
    actors: List[str]
    targets: List[str]
    actions: List[str]
    causal_role: str
    narrative_act: str
    state_tags: List[str]
    event_roles: List[str]
    causal_schemas: List[str]
    narrative_acts: List[str]
    transition_to_next: LLMTransition
    confidence: float


@dataclass
class LLMConstraintStep:
    chunk_id: int
    chunk_text: str
    current_state: str
    preferred_next_states: List[str]
    forbidden_next_states: List[str]
    next_step_constraints: List[str]
    trigger_events: List[str]
    rationale: str
    confidence: float


@dataclass
class LLMGraphNode:
    node_id: str
    chunk_id: int
    summary: str
    event_roles: List[str]
    causal_schemas: List[str]
    narrative_acts: List[str]
    state_tags: List[str]


@dataclass
class LLMGraphEdge:
    source: str
    target: str
    type: str
    reason: str


@dataclass
class LLMSessionGraph:
    session_summary: str
    nodes: List[LLMGraphNode]
    edges: List[LLMGraphEdge]
    reframes: List[Dict[str, str]]
    causal_chains: List[List[str]]
    narrative_flow: List[str]


class LLMStateExtractor:
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str = "",
        backend: str = "anthropic",
        cache_dir: Path | None = None,
        chunk_sentences: int = 2,
        max_retries: int = 2,
        timeout_seconds: int = 180,
    ) -> None:
        self.model = model
        self.api_key = self._load_api_key(api_key)
        self.backend = backend
        self.cache_dir = (cache_dir or Path(".cache/llm_state_extractor")).resolve()
        self.chunk_sentences = int(chunk_sentences)
        self.max_retries = int(max_retries)
        self.timeout_seconds = int(timeout_seconds)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cursor_lock_path = self.cache_dir / "cursor_agent.lock"

    @staticmethod
    def _load_api_key(key_arg: str = "") -> str:
        if key_arg:
            return key_arg
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if env_key:
            return env_key
        keys_path = Path.home() / ".config" / "lu" / "keys.json"
        if keys_path.exists():
            try:
                keys = json.loads(keys_path.read_text(encoding="utf-8"))
                for name in ("lu_key", "anthropic_milla", "anthropic_claude_code_main"):
                    val = keys.get(name, "")
                    if isinstance(val, str) and val.startswith("sk-ant-"):
                        return val
                for section in ("anthropic", "anthropic_milla", "anthropic_claude_code_main"):
                    sec = keys.get(section, {})
                    if isinstance(sec, dict):
                        for name in ("lu_key", "api_key", "key"):
                            val = sec.get(name, "")
                            if isinstance(val, str) and val.startswith("sk-ant-"):
                                return val
            except Exception:
                return ""
        return ""

    def available(self) -> bool:
        if self.backend == "anthropic":
            return bool(self.api_key)
        if self.backend == "cursor-agent":
            return self._cursor_agent_available()
        if self.backend == "auto":
            return bool(self.api_key) or self._cursor_agent_available()
        return False

    def extract_chunk_states(self, text: str, session_id: str = "") -> List[LLMChunkState]:
        chunks = sentence_chunks(text, chunk_size=self.chunk_sentences)
        if not chunks:
            return []
        cached = self._load_cached_chunk_states(text=text, chunks=chunks, session_id=session_id, kind="chunk_states")
        if cached is not None:
            return cached
        if not self.available():
            raise RuntimeError(f"LLM backend `{self.backend}` not available for state extraction.")
        prompt = self._build_chunk_prompt(chunks)
        last_error: Exception | None = None
        states: List[LLMChunkState] | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                payload = self._call_llm(prompt=prompt, max_tokens=2400)
                states = self._parse_chunk_state_payload(payload, chunks)
                break
            except Exception as exc:
                last_error = exc
        if states is None:
            states = self._fallback_chunk_states(chunks, last_error)
        self._write_cached_chunk_states(text=text, chunks=chunks, session_id=session_id, states=states, kind="chunk_states")
        return states

    def extract_constraint_steps(self, text: str, session_id: str = "") -> List[LLMConstraintStep]:
        chunks = sentence_chunks(text, chunk_size=self.chunk_sentences)
        if not chunks:
            return []
        cached = self._load_cached_constraint_steps(text=text, chunks=chunks, session_id=session_id, kind="constraint_steps")
        if cached is not None:
            return cached
        if not self.available():
            raise RuntimeError(f"LLM backend `{self.backend}` not available for constraint extraction.")
        prompt = self._build_constraint_prompt(chunks)
        last_error: Exception | None = None
        steps: List[LLMConstraintStep] | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                payload = self._call_llm(prompt=prompt, max_tokens=2600)
                steps = self._parse_constraint_payload(payload, chunks)
                break
            except Exception as exc:
                last_error = exc
        if steps is None:
            steps = self._fallback_constraint_steps(chunks, last_error)
        self._write_cached_constraint_steps(text=text, chunks=chunks, session_id=session_id, steps=steps, kind="constraint_steps")
        return steps

    def build_session_graph(self, states: Sequence[LLMChunkState]) -> LLMSessionGraph:
        nodes = [
            LLMGraphNode(
                node_id=f"n{state.chunk_id}",
                chunk_id=state.chunk_id,
                summary=state.summary,
                event_roles=list(state.event_roles),
                causal_schemas=list(state.causal_schemas),
                narrative_acts=list(state.narrative_acts),
                state_tags=list(state.state_tags),
            )
            for state in states
        ]
        edges = []
        reframes = []
        narrative_flow = []
        for left, right in zip(states[:-1], states[1:]):
            edges.append(
                LLMGraphEdge(
                    source=f"n{left.chunk_id}",
                    target=f"n{right.chunk_id}",
                    type=left.transition_to_next.type,
                    reason=left.transition_to_next.reason,
                )
            )
            left_entities = set(left.actors + left.targets)
            right_entities = set(right.actors + right.targets)
            overlap = sorted(left_entities.intersection(right_entities))
            if overlap and left.transition_to_next.type == "reframe":
                reframes.append(
                    {
                        "entity": overlap[0],
                        "source": f"n{left.chunk_id}",
                        "target": f"n{right.chunk_id}",
                        "type": "same_entity_role_change",
                    }
                )
        for state in states:
            narrative_flow.extend(state.narrative_acts[:2] or [state.narrative_act])
        narrative_flow = list(dict.fromkeys(narrative_flow))
        causal_chains = self._derive_causal_chains(states)
        session_summary = " ".join(state.summary for state in states[:3]).strip()
        return LLMSessionGraph(
            session_summary=session_summary[:400],
            nodes=nodes,
            edges=edges,
            reframes=reframes,
            causal_chains=causal_chains,
            narrative_flow=narrative_flow,
        )

    def _derive_causal_chains(self, states: Sequence[LLMChunkState]) -> List[List[str]]:
        chains: List[List[str]] = []
        current: List[str] = []
        causal_types = {"cause_to_decision", "decision_to_effect", "decision_to_implementation", "turn_to_resolution"}
        for state in states[:-1]:
            if state.transition_to_next.type in causal_types:
                if not current:
                    current.append(f"n{state.chunk_id}")
                current.append(f"n{state.chunk_id + 1}")
            elif len(current) >= 2:
                chains.append(current)
                current = []
        if len(current) >= 2:
            chains.append(current)
        return chains

    def _call_anthropic(self, prompt: str, max_tokens: int) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_MESSAGES_URL,
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read())
                raw = result["content"][0]["text"].strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                return raw
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Anthropic request failed after retries: {last_error}")

    def _cursor_agent_available(self) -> bool:
        cursor_bin = shutil.which("cursor")
        if not cursor_bin:
            return False
        try:
            result = subprocess.run(
                ["/bin/zsh", "-lic", "cursor agent status"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception:
            return False
        status_text = f"{result.stdout}\n{result.stderr}".lower()
        return "not logged in" not in status_text and ("logged in" in status_text or result.returncode == 0)

    def _call_cursor_agent(self, prompt: str) -> str:
        cursor_bin = shutil.which("cursor")
        if not cursor_bin:
            raise RuntimeError("`cursor` binary not found for cursor-agent backend.")
        command = (
            "cursor agent --print --mode ask --output-format text -f "
            + shlex.quote(prompt)
        )
        with self.cursor_lock_path.open("w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                result = subprocess.run(
                    ["/bin/zsh", "-lic", command],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        if result.returncode != 0 or "not logged in" in combined.lower():
            raise RuntimeError(f"Cursor agent request failed: {combined.strip()[:500]}")
        raw = result.stdout.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return raw

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        if self.backend == "cursor-agent":
            return self._call_cursor_agent(prompt)
        if self.backend == "anthropic":
            return self._call_anthropic(prompt, max_tokens)
        if self.backend == "auto":
            if self.api_key:
                return self._call_anthropic(prompt, max_tokens)
            if self._cursor_agent_available():
                return self._call_cursor_agent(prompt)
            raise RuntimeError("No available LLM backend: neither Anthropic key nor logged-in Cursor agent.")
        raise RuntimeError(f"Unsupported LLM backend: {self.backend}")

    def _build_chunk_prompt(self, chunks: Sequence[str]) -> str:
        numbered_chunks = "\n".join(f"[{idx}] {chunk}" for idx, chunk in enumerate(chunks))
        return (
            "You are extracting state transitions from a session.\n"
            "Return ONLY valid JSON.\n"
            "Produce an object with one key: \"chunks\".\n"
            "The value must be a list of chunk objects in order.\n"
            "For each chunk object include exactly these fields:\n"
            "- chunk_id (integer)\n"
            "- summary (string)\n"
            "- actors (array of strings)\n"
            "- targets (array of strings)\n"
            "- actions (array of strings)\n"
            "- causal_role (string)\n"
            "- narrative_act (string)\n"
            "- state_tags (array of strings)\n"
            "- event_roles (array of strings)\n"
            "- causal_schemas (array of strings)\n"
            "- narrative_acts (array of strings)\n"
            "- transition_to_next (object with keys: type, reason)\n"
            "- confidence (number in [0,1])\n"
            "Use compact canonical labels where possible.\n"
            "Allowed causal_role labels: cause, effect, condition, deliberation, resolution, background.\n"
            "Allowed narrative_act labels: opening, setup, complication, turn, commitment, resolution, closing, inquiry.\n"
            "Allowed transition_to_next.type labels: persistence, reframe, cause_to_decision, decision_to_effect, decision_to_implementation, question_to_resolution, setup_to_turn, turn_to_resolution, topic_shift, parallel_branch.\n"
            "If unsure, use conservative defaults.\n\n"
            f"Chunks:\n{numbered_chunks}\n"
        )

    def _build_constraint_prompt(self, chunks: Sequence[str]) -> str:
        numbered_chunks = "\n".join(f"[{idx}] {chunk}" for idx, chunk in enumerate(chunks))
        return (
            "You are extracting a local transition constraint object from a session.\n"
            "Return ONLY valid JSON.\n"
            "Produce an object with one key: \"constraints\".\n"
            "The value must be a list of objects in chunk order.\n"
            "For each object include exactly these fields:\n"
            "- chunk_id (integer)\n"
            "- current_state (string)\n"
            "- preferred_next_states (array of strings)\n"
            "- forbidden_next_states (array of strings)\n"
            "- next_step_constraints (array of strings)\n"
            "- trigger_events (array of strings)\n"
            "- rationale (string)\n"
            "- confidence (number in [0,1])\n"
            "Use compact labels. Focus on structural next-step constraints, not full semantic paraphrase.\n"
            "Allowed trigger families include turn, reframe, topic_shift, question, resolution, cause_to_decision, tone_flip, persistence.\n"
            "Keep arrays short and canonical.\n\n"
            f"Chunks:\n{numbered_chunks}\n"
        )

    def _parse_chunk_state_payload(self, raw_payload: str, chunks: Sequence[str]) -> List[LLMChunkState]:
        data = json.loads(self._extract_first_json_object(raw_payload))
        if not isinstance(data, dict) or "chunks" not in data or not isinstance(data["chunks"], list):
            raise ValueError("LLM chunk payload missing `chunks` list.")
        parsed: List[LLMChunkState] = []
        for expected_id, (item, chunk_text) in enumerate(zip(data["chunks"], chunks)):
            parsed.append(self._normalize_chunk_state(item, chunk_text, expected_id))
        if len(parsed) != len(chunks):
            raise ValueError("LLM chunk payload length mismatch.")
        return parsed

    def _parse_constraint_payload(self, raw_payload: str, chunks: Sequence[str]) -> List[LLMConstraintStep]:
        data = json.loads(self._extract_first_json_object(raw_payload))
        if not isinstance(data, dict) or "constraints" not in data or not isinstance(data["constraints"], list):
            raise ValueError("LLM constraint payload missing `constraints` list.")
        parsed: List[LLMConstraintStep] = []
        for expected_id, (item, chunk_text) in enumerate(zip(data["constraints"], chunks)):
            parsed.append(self._normalize_constraint_step(item, chunk_text, expected_id))
        if len(parsed) != len(chunks):
            raise ValueError("LLM constraint payload length mismatch.")
        return parsed

    def _extract_first_json_object(self, raw_payload: str) -> str:
        raw_payload = raw_payload.strip()
        if raw_payload.startswith("{") and raw_payload.endswith("}"):
            return raw_payload
        start = raw_payload.find("{")
        if start < 0:
            raise ValueError("No JSON object found in LLM payload.")
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(raw_payload)):
            char = raw_payload[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return raw_payload[start : idx + 1]
        raise ValueError("Incomplete JSON object in LLM payload.")

    def _fallback_chunk_states(self, chunks: Sequence[str], error: Exception | None) -> List[LLMChunkState]:
        fallback_states: List[LLMChunkState] = []
        for idx, chunk_text in enumerate(chunks):
            tokens = tokenize(chunk_text)
            lead = tokens[:3] or ["context"]
            summary = " ".join(tokens[:12]) if tokens else chunk_text[:120]
            fallback_states.append(
                LLMChunkState(
                    chunk_id=idx,
                    chunk_text=chunk_text,
                    summary=summary,
                    actors=lead[:1] or ["speaker"],
                    targets=lead[1:2] or ["topic"],
                    actions=lead[2:3] or ["describe"],
                    causal_role="background",
                    narrative_act="setup",
                    state_tags=list(dict.fromkeys(lead + ["fallback"]))[:4],
                    event_roles=[
                        f"actor:{(lead[:1] or ['speaker'])[0]}",
                        f"target:{(lead[1:2] or ['topic'])[0]}",
                        f"action:{(lead[2:3] or ['describe'])[0]}",
                    ],
                    causal_schemas=["persistence"],
                    narrative_acts=["setup"],
                    transition_to_next=LLMTransition(
                        type="persistence",
                        reason=f"fallback_due_to_invalid_llm_output:{type(error).__name__ if error else 'unknown'}",
                    ),
                    confidence=0.05,
                )
            )
        return fallback_states

    def _fallback_constraint_steps(self, chunks: Sequence[str], error: Exception | None) -> List[LLMConstraintStep]:
        fallback_steps: List[LLMConstraintStep] = []
        for idx, chunk_text in enumerate(chunks):
            tokens = tokenize(chunk_text)
            lead = tokens[:3] or ["context"]
            fallback_steps.append(
                LLMConstraintStep(
                    chunk_id=idx,
                    chunk_text=chunk_text,
                    current_state=(lead[0] if lead else "context"),
                    preferred_next_states=["persistence"],
                    forbidden_next_states=[],
                    next_step_constraints=["maintain_local_context"],
                    trigger_events=["persistence"],
                    rationale=f"fallback_due_to_invalid_llm_output:{type(error).__name__ if error else 'unknown'}",
                    confidence=0.05,
                )
            )
        return fallback_steps

    def _normalize_chunk_state(self, item: Dict[str, object], chunk_text: str, expected_id: int) -> LLMChunkState:
        if not isinstance(item, dict):
            raise ValueError("Chunk state entry must be a JSON object.")

        def _list_field(name: str, limit: int) -> List[str]:
            value = item.get(name, [])
            if isinstance(value, list):
                out = [str(entry).strip().lower() for entry in value if str(entry).strip()]
            elif isinstance(value, str) and value.strip():
                out = [value.strip().lower()]
            else:
                out = []
            deduped = list(dict.fromkeys(out))
            return deduped[:limit]

        def _str_field(name: str, default: str) -> str:
            value = item.get(name, default)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
            return default

        def _confidence() -> float:
            value = item.get("confidence", 0.5)
            try:
                score = float(value)
            except Exception:
                score = 0.5
            return min(max(score, 0.0), 1.0)

        transition_obj = item.get("transition_to_next", {})
        if not isinstance(transition_obj, dict):
            transition_obj = {}
        transition = LLMTransition(
            type=str(transition_obj.get("type", "persistence")).strip().lower() or "persistence",
            reason=str(transition_obj.get("reason", "")).strip(),
        )

        chunk_id = int(item.get("chunk_id", expected_id))
        if chunk_id != expected_id:
            chunk_id = expected_id

        actors = _list_field("actors", 4)
        targets = _list_field("targets", 4)
        actions = _list_field("actions", 4)
        state_tags = _list_field("state_tags", 6)
        event_roles = _list_field("event_roles", 8)
        causal_schemas = _list_field("causal_schemas", 6)
        narrative_acts = _list_field("narrative_acts", 6)
        if not event_roles:
            if actors:
                event_roles.append(f"actor:{actors[0]}")
            if targets:
                event_roles.append(f"target:{targets[0]}")
            if actions:
                event_roles.append(f"action:{actions[0]}")
        return LLMChunkState(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            summary=str(item.get("summary", chunk_text[:160])).strip(),
            actors=actors,
            targets=targets,
            actions=actions,
            causal_role=_str_field("causal_role", "background"),
            narrative_act=_str_field("narrative_act", "setup"),
            state_tags=state_tags,
            event_roles=event_roles,
            causal_schemas=causal_schemas,
            narrative_acts=narrative_acts or [_str_field("narrative_act", "setup")],
            transition_to_next=transition,
            confidence=_confidence(),
        )

    def _normalize_constraint_step(
        self,
        item: Dict[str, object],
        chunk_text: str,
        expected_id: int,
    ) -> LLMConstraintStep:
        if not isinstance(item, dict):
            raise ValueError("Constraint step entry must be a JSON object.")

        def _list_field(name: str, limit: int) -> List[str]:
            value = item.get(name, [])
            if isinstance(value, list):
                out = [str(entry).strip().lower() for entry in value if str(entry).strip()]
            elif isinstance(value, str) and value.strip():
                out = [value.strip().lower()]
            else:
                out = []
            return list(dict.fromkeys(out))[:limit]

        def _str_field(name: str, default: str) -> str:
            value = item.get(name, default)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
            return default

        def _confidence() -> float:
            value = item.get("confidence", 0.5)
            try:
                score = float(value)
            except Exception:
                score = 0.5
            return min(max(score, 0.0), 1.0)

        chunk_id = int(item.get("chunk_id", expected_id))
        if chunk_id != expected_id:
            chunk_id = expected_id

        preferred = _list_field("preferred_next_states", 5) or ["persistence"]
        forbidden = _list_field("forbidden_next_states", 5)
        constraints = _list_field("next_step_constraints", 6) or ["maintain_local_context"]
        triggers = _list_field("trigger_events", 4) or ["persistence"]
        return LLMConstraintStep(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            current_state=_str_field("current_state", "context"),
            preferred_next_states=preferred,
            forbidden_next_states=forbidden,
            next_step_constraints=constraints,
            trigger_events=triggers,
            rationale=str(item.get("rationale", chunk_text[:160])).strip(),
            confidence=_confidence(),
        )

    def _cache_key(self, text: str, chunks: Sequence[str], session_id: str, kind: str) -> str:
        digest = hashlib.sha256()
        digest.update(kind.encode("utf-8"))
        digest.update(self.backend.encode("utf-8"))
        digest.update(self.model.encode("utf-8"))
        digest.update(str(self.chunk_sentences).encode("utf-8"))
        digest.update(text.encode("utf-8"))
        for chunk in chunks:
            digest.update(chunk.encode("utf-8"))
        return digest.hexdigest()

    def _cache_path(self, text: str, chunks: Sequence[str], session_id: str, kind: str) -> Path:
        return self.cache_dir / f"{self._cache_key(text, chunks, session_id, kind)}.json"

    def _load_cached_chunk_states(
        self,
        text: str,
        chunks: Sequence[str],
        session_id: str,
        kind: str,
    ) -> Optional[List[LLMChunkState]]:
        path = self._cache_path(text, chunks, session_id, kind)
        if not path.exists():
            return None

    def _load_cached_constraint_steps(
        self,
        text: str,
        chunks: Sequence[str],
        session_id: str,
        kind: str,
    ) -> Optional[List[LLMConstraintStep]]:
        path = self._cache_path(text, chunks, session_id, kind)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [
                LLMConstraintStep(
                    chunk_id=int(item["chunk_id"]),
                    chunk_text=str(item["chunk_text"]),
                    current_state=str(item["current_state"]),
                    preferred_next_states=list(item["preferred_next_states"]),
                    forbidden_next_states=list(item["forbidden_next_states"]),
                    next_step_constraints=list(item["next_step_constraints"]),
                    trigger_events=list(item["trigger_events"]),
                    rationale=str(item["rationale"]),
                    confidence=float(item["confidence"]),
                )
                for item in payload["constraints"]
            ]
        except Exception:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [
                LLMChunkState(
                    chunk_id=int(item["chunk_id"]),
                    chunk_text=str(item["chunk_text"]),
                    summary=str(item["summary"]),
                    actors=list(item["actors"]),
                    targets=list(item["targets"]),
                    actions=list(item["actions"]),
                    causal_role=str(item["causal_role"]),
                    narrative_act=str(item["narrative_act"]),
                    state_tags=list(item["state_tags"]),
                    event_roles=list(item["event_roles"]),
                    causal_schemas=list(item["causal_schemas"]),
                    narrative_acts=list(item["narrative_acts"]),
                    transition_to_next=LLMTransition(**item["transition_to_next"]),
                    confidence=float(item["confidence"]),
                )
                for item in payload["chunks"]
            ]
        except Exception:
            return None

    def _write_cached_chunk_states(
        self,
        text: str,
        chunks: Sequence[str],
        session_id: str,
        states: Sequence[LLMChunkState],
        kind: str,
    ) -> None:
        path = self._cache_path(text, chunks, session_id, kind)
        payload = {"chunks": [asdict(state) for state in states]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_cached_constraint_steps(
        self,
        text: str,
        chunks: Sequence[str],
        session_id: str,
        steps: Sequence[LLMConstraintStep],
        kind: str,
    ) -> None:
        path = self._cache_path(text, chunks, session_id, kind)
        payload = {"constraints": [asdict(step) for step in steps]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def chunk_states_to_session_graph(states: Sequence[LLMChunkState]) -> LLMSessionGraph:
    extractor = LLMStateExtractor(api_key="")
    return extractor.build_session_graph(states)


def _tone_markers_from_llm_state(state: LLMChunkState) -> List[str]:
    blob = " ".join(
        [
            state.summary,
            state.causal_role,
            state.narrative_act,
            state.transition_to_next.type,
            " ".join(state.state_tags),
            " ".join(state.narrative_acts),
        ]
    ).lower()
    tones: List[str] = []
    if any(token in blob for token in ("question", "inquiry", "uncertain", "maybe", "consider")):
        tones.append("uncertain")
    if any(token in blob for token in ("resolve", "resolution", "decide", "commit", "therefore")):
        tones.append("assertive")
    if any(token in blob for token in ("turn", "reframe", "contrast", "however", "but", "topic_shift")):
        tones.append("contrastive")
    if any(token in blob for token in ("reflect", "reflection", "background", "setup")):
        tones.append("reflective")
    if any(token in blob for token in ("risk", "problem", "failure", "concern", "conflict")):
        tones.append("tense")
    if any(token in blob for token in ("benefit", "opportunity", "success", "progress", "resolution")):
        tones.append("positive")
    return list(dict.fromkeys(tones))[:4]


def llm_chunk_states_to_state_fields(states: Sequence[LLMChunkState]) -> List[Dict[str, object]]:
    mapped: List[Dict[str, object]] = []
    for state in states:
        semantic_keys = list(dict.fromkeys(state.actors + state.targets + state.state_tags))[:6] or ["misc"]
        relation_keys = list(dict.fromkeys(state.actions + [state.transition_to_next.type]))[:6] or ["context"]
        causal_markers = list(dict.fromkeys([state.causal_role] + state.causal_schemas))[:4]
        turn_markers = list(dict.fromkeys([state.narrative_act] + state.narrative_acts))[:4]
        tone_markers = _tone_markers_from_llm_state(state)
        flags = []
        label_blob = " ".join(relation_keys + turn_markers + state.state_tags).lower()
        if any(key in label_blob for key in ("decision", "commit")):
            flags.append("DECISION")
        if any(key in label_blob for key in ("turn", "pivot", "reframe")):
            flags.append("PIVOT")
        if any(key in label_blob for key in ("principle", "core")):
            flags.append("CORE")
        mapped.append(
            {
                "chunk_text": state.chunk_text,
                "tokens": tokenize(state.chunk_text),
                "semantic_keys": semantic_keys,
                "relation_keys": relation_keys,
                "flags": flags,
                "emotions": tone_markers,
                "causal_markers": causal_markers,
                "turn_markers": turn_markers,
                "event_roles": list(state.event_roles),
                "causal_schemas": list(state.causal_schemas),
                "narrative_acts": list(state.narrative_acts),
                "weight": 1.0 + 0.6 * state.confidence + 0.05 * len(semantic_keys) + 0.05 * len(relation_keys),
            }
        )
    return mapped


def llm_constraint_steps_to_state_fields(steps: Sequence[LLMConstraintStep]) -> List[Dict[str, object]]:
    mapped: List[Dict[str, object]] = []
    for step in steps:
        semantic_keys = [step.current_state] + [f"state::{step.current_state}"] + step.trigger_events[:2]
        semantic_keys = list(dict.fromkeys(item for item in semantic_keys if item))[:6] or ["constraint"]
        relation_keys = list(
            dict.fromkeys(
                [f"allow::{item}" for item in step.preferred_next_states[:3]]
                + [f"require::{item}" for item in step.next_step_constraints[:3]]
                + [f"block::{item}" for item in step.forbidden_next_states[:2]]
            )
        )[:6] or ["allow::persistence"]
        flags: List[str] = []
        if step.forbidden_next_states:
            flags.append("CONSTRAINT")
        if any(tag in step.trigger_events for tag in ("reframe", "topic_shift", "turn")):
            flags.append("PIVOT")
        if any(tag in step.trigger_events for tag in ("resolution", "cause_to_decision")):
            flags.append("DECISION")
        mapped.append(
            {
                "chunk_text": step.chunk_text,
                "tokens": tokenize(step.chunk_text),
                "semantic_keys": semantic_keys,
                "relation_keys": relation_keys,
                "flags": flags,
                "emotions": [],
                "causal_markers": list(dict.fromkeys(step.next_step_constraints[:3]))[:4],
                "turn_markers": list(dict.fromkeys([f"trigger::{item}" for item in step.trigger_events[:3]]))[:4],
                "event_roles": [
                    f"state::{step.current_state}",
                    *[f"prefer::{item}" for item in step.preferred_next_states[:2]],
                    *[f"forbid::{item}" for item in step.forbidden_next_states[:2]],
                    *[f"trigger::{item}" for item in step.trigger_events[:2]],
                ][:8],
                "causal_schemas": list(dict.fromkeys([f"require::{item}" for item in step.next_step_constraints[:4]]))[:6],
                "narrative_acts": list(dict.fromkeys([f"trigger::{item}" for item in step.trigger_events[:3]] + [f"state::{step.current_state}"]))[:6],
                "weight": 1.0 + 0.7 * step.confidence + 0.08 * len(step.preferred_next_states) + 0.05 * len(step.forbidden_next_states),
            }
        )
    return mapped
