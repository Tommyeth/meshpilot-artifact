"""Model prompt, response schema and released payload construction."""
from __future__ import annotations

import hashlib
from typing import Any

from .models import CandidatePlan, TaskIntent


AGENT_PROMPT_VERSION = "meshpilot-policy-v2"

POLICY_SYSTEM_INSTRUCTION = (
    "You are a resource orchestration policy agent. Infer whether the operator primarily values "
    "latency, cost, resilience, or a balance. Select only one supplied plan_id. Never invent IDs, "
    "resource quantities, commands, or hidden facts. Keep rationale to one sentence under 160 "
    "characters. Abstain if the intent cannot be satisfied."
)

AGENT_PROMPT_SHA256 = hashlib.sha256(POLICY_SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest()

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["select", "abstain"]},
        "plan_id": {"type": ["string", "null"]},
        "rationale": {"type": "string", "maxLength": 500},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "inferred_profile": {
            "type": "string",
            "enum": ["balanced", "latency", "cost", "resilience"],
        },
    },
    "required": ["action", "plan_id", "rationale", "confidence", "inferred_profile"],
    "additionalProperties": False,
}

GENERATION_CONFIG = {
    "system_instruction": POLICY_SYSTEM_INSTRUCTION,
    "response_mime_type": "application/json",
    "response_json_schema": DECISION_SCHEMA,
    "temperature": 0.0,
    "candidate_count": 1,
    "max_output_tokens": 512,
    "thinking_config": {"thinking_budget": 0},
}


def build_model_payload(task: TaskIntent, candidates: list[CandidatePlan]) -> dict[str, Any]:
    """Construct the released JSON payload without invoking a model."""
    return {
        "operator_intent": task.operator_intent,
        "hard_requirements": {
            "deadline_ms": task.deadline_ms,
            "deadline_is_hard": task.deadline_is_hard,
            "max_cost_usd": task.max_cost_usd,
            "min_bandwidth_mbps": task.min_bandwidth_mbps,
        },
        "candidates": [
            candidate.model_dump(
                include={
                    "plan_id",
                    "node_id",
                    "predicted_completion_ms",
                    "predicted_cost_usd",
                    "dominant_share_after",
                    "risk_score",
                    "reserved_bandwidth_mbps",
                }
            )
            for candidate in candidates
        ],
    }
