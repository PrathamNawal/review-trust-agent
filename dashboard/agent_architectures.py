"""Registry for interchangeable investigation architectures.

Each architecture accepts the same review/config dictionaries and emits the
same trace-event dictionaries. The current implementation registers Agno; new
architectures can be added without changing the Streamlit pages.
"""

from collections.abc import Iterator

from dashboard.contracts import AgentConfig, ArchitectureRunner, Review, TraceEvent


_ARCHITECTURES = {}


def register_architecture(name: str, runner: ArchitectureRunner):
    """Register or replace an architecture runner under a stable config key."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Architecture name cannot be empty")
    _ARCHITECTURES[key] = runner


def available_architectures():
    return tuple(_ARCHITECTURES)


def run_investigation_stream(review: Review, config: AgentConfig) -> Iterator[TraceEvent]:
    architecture = config.get("architecture", "agno")
    runner = _ARCHITECTURES.get(architecture)
    if runner is None:
        yield {
            "type": "error",
            "message": f"Unknown agent architecture: {architecture}. "
                       f"Available: {', '.join(available_architectures()) or '(none)'}.",
        }
        return
    yield from runner(review, config)
