"""Convert filtered Q&A pairs to Harmony format for gpt-oss-20b fine-tuning."""

from __future__ import annotations

from src.common.config import load_prompt
from src.common.models import (
    HarmonyMessage,
    HarmonyExample,
    TrainingExample,
    GeneratedQuestion,
    ExpertResponse,
    MultiTurnTurn,
)

PERSONA = None


def _get_persona() -> str:
    global PERSONA
    if PERSONA is None:
        PERSONA = load_prompt("system_architect_persona")
    return PERSONA


def single_turn_to_harmony(
    question: GeneratedQuestion,
    response: ExpertResponse,
    quality_score: float = 0.0,
) -> TrainingExample:
    """Convert a single-turn Q&A pair to Harmony format.

    Harmony format for gpt-oss-20b:
    - developer: system instructions (persona)
    - user: the question
    - assistant (analysis channel): chain-of-thought reasoning
    - assistant (final channel): structured answer
    """
    harmony = HarmonyExample(
        messages=[
            HarmonyMessage(role="developer", content=_get_persona()),
            HarmonyMessage(role="user", content=question.question),
            HarmonyMessage(
                role="assistant",
                channel="analysis",
                content=response.chain_of_thought,
            ),
            HarmonyMessage(
                role="assistant",
                channel="final",
                content=response.response,
            ),
        ]
    )

    return TrainingExample(
        harmony=harmony,
        primitive=question.primitive,
        domain=question.domain,
        complexity=question.complexity,
        quality_score=quality_score,
    )


def multi_turn_to_harmony(
    turns: list[MultiTurnTurn],
    primitive: str,
    domain: str,
    quality_score: float = 0.0,
) -> TrainingExample:
    """Convert a multi-turn conversation to Harmony format.

    Each turn gets:
    - user message
    - assistant analysis (chain-of-thought)
    - assistant final (response)
    """
    messages = [HarmonyMessage(role="developer", content=_get_persona())]

    for turn in turns:
        messages.extend([
            HarmonyMessage(role="user", content=turn.user),
            HarmonyMessage(
                role="assistant",
                channel="analysis",
                content=turn.chain_of_thought,
            ),
            HarmonyMessage(
                role="assistant",
                channel="final",
                content=turn.response,
            ),
        ])

    harmony = HarmonyExample(messages=messages)

    return TrainingExample(
        harmony=harmony,
        primitive=primitive,
        domain=domain,
        complexity="multi_turn",
        quality_score=quality_score,
    )
