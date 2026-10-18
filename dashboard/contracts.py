"""Stable data contracts shared by agent architectures and the Streamlit UI."""

from collections.abc import Iterator
from typing import Literal, Protocol, TypedDict


class Review(TypedDict):
    review_id: int
    user_id: str
    prod_id: str
    rating: float
    date: str
    review_text: str


class AgentConfig(TypedDict, total=False):
    version: str
    architecture: str
    system_prompt: str
    provider: str
    model: str
    temperature: float
    confidence_threshold: float
    jev_method: str
    jev_threshold: float
    jev_prompts: dict[str, str]
    jev_evidence_sources: list[str]


class TraceEvent(TypedDict, total=False):
    type: Literal["turn", "tool_call", "final", "error"]
    turn: int
    tool: str
    input: dict
    result: object
    predicted_filtered: bool
    confidence: float
    primary_signal: str
    reasoning: str
    turns_used: int
    _incomplete: bool
    message: str


class ArchitectureRunner(Protocol):
    def __call__(self, review: Review, config: AgentConfig) -> Iterator[TraceEvent]: ...
