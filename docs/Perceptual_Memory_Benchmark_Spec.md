# Perceptual Memory Benchmark Spec

## Goal
`PMS-Bench` evaluates whether a compressed memory system behaves like a perceptual memory system rather than a plain retrieval index.

The benchmark is designed for systems whose target pipeline resembles:

`raw experience -> compressed memory object -> cue-driven recovery -> perceptual reconstruction`

This benchmark is intended to compare:
- `constraint-edge` style symbolic memory systems
- `native volume` / `archive-native` K3M systems
- retrieval baselines such as AAAK or raw vector search

## Design Principle
The benchmark should reward systems that:
- recover the right event, cause, turn, and constraint structure
- preserve scene-level perceptual attributes such as tone and mode
- operate under explicit memory budgets
- remain structurally stable when relation or chunk structure is perturbed

The benchmark should not reward only session-level retrieval ranking.

## Task Families
Each benchmark item belongs to one primary `task_type`.

### 1. Event Recall
Recover the correct event or episode from memory.

Typical questions:
- Why did we abandon plan A?
- What happened in the replacement earrings case?

Primary targets:
- gold session
- gold span
- gold entities

### 2. Causal Recall
Recover the main causal chain or dominant causal explanation.

Typical questions:
- What caused the later decision?
- Why was repair tracking introduced?

Primary targets:
- gold causal chain
- primary cause
- supporting entities

### 3. Turning Point Recall
Recover the key reframe, pivot, or state transition boundary.

Typical questions:
- When did the discussion shift direction?
- Which turn changed the plan?

Primary targets:
- turning-point span
- pre-state and post-state

### 4. Constraint Recall
Recover explicit next-step constraints and compare them with actual evolution.

Typical questions:
- What was allowed next?
- What was explicitly blocked?
- Did the actual next step satisfy or violate the constraint?

Primary targets:
- allowed set
- forbidden set
- required set
- actual next
- status in `{satisfied, violated, outside}`

### 5. Cue-Based Perceptual Recall
Recover memory from weak or partial cues rather than a fully specified question.

Typical cues:
- a phrase
- an object
- a person
- a weak trigger

Primary targets:
- memory cluster or event span
- scene type
- supporting entities

## Runtime Document Format
`PMS-Bench` uses versioned runtime documents rather than bare JSON lists.

Canonical item document:

```json
{
  "benchmark_name": "PMS-Bench",
  "format_version": "2.0",
  "items": [
    {
      "item_id": "pm_0001",
      "task_type": "constraint_recall",
      "question": "What was allowed next and what actually happened?",
      "weak_cues": ["repair history", "watch", "battery replacement"],
      "gold_sessions": ["sess_123"],
      "gold_spans": [
        {
          "session_id": "sess_123",
          "start_turn": 4,
          "end_turn": 7
        }
      ],
      "gold_entities": ["watch", "battery", "repair"],
      "gold_constraints": {
        "allowed": ["add_watch_record", "track_repair_need"],
        "forbidden": ["drop_watch_context"],
        "required": ["preserve_repair_history"],
        "actual_next": "add_watch_record",
        "status": "satisfied"
      },
      "gold_causal_chain": [],
      "gold_turning_point": null,
      "gold_scene": {
        "tone": "practical",
        "mode": "inventory_management",
        "uncertainty": 0.15
      }
    }
  ]
}
```

Legacy list payloads may still be read during migration, but `2.0` documents are the canonical contract.

## Item Format
Each benchmark item has this conceptual structure:

```json
{
  "item_id": "pm_0001",
  "task_type": "constraint_recall",
  "question": "What was allowed next and what actually happened?",
  "weak_cues": ["repair history", "watch", "battery replacement"],
  "gold_sessions": ["sess_123"],
  "gold_spans": [
    {
      "session_id": "sess_123",
      "start_turn": 4,
      "end_turn": 7
    }
  ],
  "gold_entities": ["watch", "battery", "repair"],
  "gold_constraints": {
    "allowed": ["add_watch_record", "track_repair_need"],
    "forbidden": ["drop_watch_context"],
    "required": ["preserve_repair_history"],
    "actual_next": "add_watch_record",
    "status": "satisfied"
  },
  "gold_causal_chain": [],
  "gold_turning_point": null,
  "gold_scene": {
    "tone": "practical",
    "mode": "inventory_management",
    "uncertainty": 0.15
  }
}
```

## Prediction Format
Canonical prediction document:

```json
{
  "benchmark_name": "PMS-Bench",
  "format_version": "2.0",
  "protocol_default": "fixed_budget",
  "predictions": [
    {
      "item_id": "pm_0001",
      "pred_sessions": ["sess_123", "sess_088"],
      "pred_spans": [
        {
          "session_id": "sess_123",
          "start_turn": 5,
          "end_turn": 7,
          "score": 0.91
        }
      ],
      "pred_entities": ["watch", "battery", "repair"],
      "pred_constraints": {
        "allowed": ["add_watch_record", "track_repair_need"],
        "forbidden": ["drop_watch_context"],
        "required": ["preserve_repair_history"],
        "actual_next": "add_watch_record",
        "status": "satisfied"
      },
      "pred_causal_chain": [],
      "pred_turning_point": null,
      "pred_scene": {
        "tone": "practical",
        "mode": "inventory_management",
        "uncertainty": 0.18
      },
      "protocol_runs": {
        "fixed_budget": {
          "pred_sessions": ["sess_123", "sess_088"]
        },
        "weak_cue": {
          "pred_sessions": ["sess_123", "sess_088"]
        },
        "progressive_recall": {
          "stages": [
            {
              "stage": "weak_cue",
              "pred_sessions": ["sess_088", "sess_123"]
            },
            {
              "stage": "refined_cue",
              "pred_sessions": ["sess_123", "sess_088"]
            },
            {
              "stage": "full_question",
              "pred_sessions": ["sess_123", "sess_088"]
            }
          ]
        },
        "structural_perturbation": {
          "chunk_shuffle": {
            "pred_sessions": ["sess_088", "sess_123"]
          },
          "relation_shuffle": {
            "pred_sessions": ["sess_123", "sess_088"]
          },
          "constraint_perturbation": {
            "pred_sessions": ["sess_123", "sess_088"]
          }
        }
      },
      "memory_stats": {
        "bytes_used": 1542,
        "latency_ms": 83,
        "archive_support_ratio": 0.031
      }
    }
  ]
}
```

Each prediction object conceptually contains:

```json
{
  "item_id": "pm_0001",
  "pred_sessions": ["sess_123", "sess_088"],
  "pred_spans": [
    {
      "session_id": "sess_123",
      "start_turn": 5,
      "end_turn": 7,
      "score": 0.91
    }
  ],
  "pred_entities": ["watch", "battery", "repair"],
  "pred_constraints": {
    "allowed": ["add_watch_record", "track_repair_need"],
    "forbidden": ["drop_watch_context"],
    "required": ["preserve_repair_history"],
    "actual_next": "add_watch_record",
    "status": "satisfied"
  },
  "pred_causal_chain": [],
  "pred_turning_point": null,
  "pred_scene": {
    "tone": "practical",
    "mode": "inventory_management",
    "uncertainty": 0.18
  },
  "memory_stats": {
    "bytes_used": 1542,
    "latency_ms": 83,
    "archive_support_ratio": 0.031
  }
}
```

## Metrics
`PMS-Bench` reports five metric groups.

### A. Retrieval Metrics
These remain useful as lower-level diagnostics.

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `MRR`

These use `gold_sessions` and `pred_sessions`.

### B. Perceptual Recovery Metrics
These measure whether the correct memory episode is recovered.

- `EventHit`
- `SpanIoU`
- `EntityF1`
- `CausalStepF1`
- `CauseHit`
- `TurningSpanIoU`
- `TurningPreStateF1`
- `TurningPostStateF1`
- `TurningPointHit`

### C. Constraint Fidelity Metrics
These are primary metrics for symbolic and constraint-based memory systems.

- `AllowedSetF1`
- `ForbiddenSetF1`
- `RequiredSetF1`
- `ActualNextAccuracy`
- `ConstraintStatusAccuracy`

### D. Scene Fidelity Metrics
These capture perceptual memory qualities not reducible to retrieval.

- `ToneAccuracy`
- `SceneModeAccuracy`
- `UncertaintyMAE`
- `SceneFidelity`

### E. Compression Utility Metrics
These evaluate memory usefulness under finite budgets.

- `AvgBytesUsed`
- `AvgLatencyMs`
- `RecoveryPerKB`
- `ArchiveSupportRatio`

## Score Aggregation
The default total score is:

`PMS = 0.22 * Retrieval + 0.22 * ConstraintFidelity + 0.22 * PerceptualRecovery + 0.14 * SceneFidelity + 0.10 * CompressionUtility + 0.10 * StructuralStability`

Where:
- `Retrieval` is computed from retrieval diagnostics on the fixed-budget run
- `ConstraintFidelity` is computed primarily on `constraint_recall` items
- `PerceptualRecovery` is computed from event / causal / turning-point recovery metrics with task-aware summaries
- `SceneFidelity` combines tone, scene mode, and normalized uncertainty error
- `CompressionUtility` is a normalized budget-aware utility score
- `StructuralStability` is defined mainly by perturbation protocols below rather than raw retrieval alone

## Protocols
### Protocol 1. Fixed Budget
All systems should be compared at an explicit budget whenever possible.

Examples:
- bytes per session
- bytes per benchmark item
- fixed archive budget per corpus slice

### Protocol 2. Weak Cue Recall
Only provide `weak_cues`, not the full question.

This tests whether compressed memory can be awakened by partial perceptual triggers.

### Protocol 3. Progressive Recall
Evaluate multi-step recovery:
1. weak cue
2. refined cue
3. full question

Measure whether the system converges toward the correct memory episode.

Suggested progressive metrics:
- `ProgressiveRecallGain`
- `ProgressiveSpanGain`
- `ProgressiveConstraintGain`

### Protocol 4. Structural Perturbation
Apply perturbations such as:
- chunk shuffle
- relation shuffle
- constraint perturbation

Measure degradation in:
- retrieval
- constraint fidelity
- archive-native structure

Suggested structural stability metrics:
- `ShuffleDegradation`
- `ConstraintConsistency`
- `CueRecallDrop`
- `TurningStateDrop`

## Annotation Guidance
Annotators should prefer:
- one dominant event span per item
- small canonical label sets for constraints
- one primary tone and one primary scene mode
- explicit `status` labels for constraint items
- tight spans around the true causal or turning boundary instead of whole-session spans
- explicit `pre_state` and `post_state` labels for turning-point items
- short canonical causal-chain steps rather than paraphrase-heavy long sentences

When multiple answers are valid:
- keep `gold_sessions` and `gold_spans` plural
- allow multiple gold entities
- keep one canonicalized constraint object if possible

## Recommended Initial Benchmark Slice
For a first benchmark release:
- 50 event items
- 50 causal items
- 50 turning-point items
- 75 constraint items
- 25 weak-cue items

This gives a minimal balanced set of `250` items.

## Intended Use
`PMS-Bench` is intended to be the main target benchmark for:
- perceptual memory systems
- K3M-style compressed memory archives
- hybrid symbolic-geometric memory models

It should replace pure retrieval-only evaluation as the primary decision surface for architecture choices.

## Results Format
Canonical result document:

```json
{
  "benchmark_name": "PMS-Bench",
  "format_version": "2.0",
  "results": {
    "item_count": 20,
    "summary": {},
    "task_summaries": {},
    "protocol_summaries": {},
    "subscores": {},
    "pms_score": 0.0
  }
}
```
