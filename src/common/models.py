"""Pydantic models for all pipeline data structures."""

from __future__ import annotations

from pydantic import BaseModel, Field


# === Part A: Extraction ===


class BlogPost(BaseModel):
    id: str = Field(description="SHA-256 of URL")
    url: str
    title: str
    author: str | None = None
    date: str | None = None
    source_blog: str
    categories: list[str] = Field(default_factory=list)
    markdown: str
    word_count: int
    crawled_at: str


# === Phase 1: Primitive Extraction ===


class PrimitiveAnnotation(BaseModel):
    post_id: str
    primary_primitive: str
    secondary_primitives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    domain_independent_lesson: str
    source_url: str


# === Phase 2: Question Generation ===


class GeneratedQuestion(BaseModel):
    id: str
    source_annotation_id: str
    primitive: str
    domain: str
    question: str
    complexity: str = Field(
        description="single_primitive | full_system | cross_domain | tradeoff | failure | multi_turn"
    )
    evol_generation: int = 0


class MultiTurnQuestion(BaseModel):
    id: str
    source_annotation_id: str
    primitive: str
    domain: str
    turns: list[str]
    complexity: str = "multi_turn"
    evol_generation: int = 0


# === Phase 3: Response Generation ===


class ExpertResponse(BaseModel):
    question_id: str
    chain_of_thought: str
    response: str


class MultiTurnTurn(BaseModel):
    user: str
    chain_of_thought: str
    response: str


class MultiTurnResponse(BaseModel):
    question_id: str
    turns: list[MultiTurnTurn]


# === Phase 4: Quality Filtering ===


class QualityScore(BaseModel):
    question_id: str
    passed_brand_check: bool
    token_count: int
    is_duplicate: bool
    nearest_neighbor_sim: float
    llm_judge_score: float
    accepted: bool


# === Final Output: Harmony Format ===


class HarmonyMessage(BaseModel):
    role: str
    content: str
    channel: str | None = None


class HarmonyExample(BaseModel):
    messages: list[HarmonyMessage]


class TrainingExample(BaseModel):
    """A complete training example with metadata for composition tracking."""
    harmony: HarmonyExample
    primitive: str
    domain: str
    complexity: str
    quality_score: float
