"""Application service for running and recording one agent investigation."""

from dashboard.agent_architectures import run_investigation_stream


def investigate_and_log(review, config, true_filtered, tracker=None):
    """Stream trace events and record one completed judgment consistently.

    Errors and turn-cap fallbacks remain visible to the UI but are not recorded
    as completed model judgments unless the runner returned a valid verdict.
    """
    if tracker is None:
        from dashboard import mlflow_tracking as tracker

    n_tool_calls = 0
    for event in run_investigation_stream(review, config):
        if event.get("type") == "tool_call":
            n_tool_calls += 1
        if (event.get("type") == "final"
                and event.get("predicted_filtered") is not None
                and not event.get("_incomplete")):
            tracker.log_judgment(
                judge_type="agent",
                config_version=config["version"],
                review_id=review["review_id"],
                true_filtered=bool(true_filtered),
                predicted_filtered=event["predicted_filtered"],
                confidence=event.get("confidence"),
                primary_signal=event.get("primary_signal"),
                reasoning=event.get("reasoning"),
                turns_used=event.get("turns_used"),
                n_tool_calls=n_tool_calls,
                provider=config.get("provider"),
                model=config.get("model"),
                temperature=config.get("temperature"),
                extra_tags={
                    "framework": config.get("architecture", "agno"),
                    "architecture": config.get("architecture", "agno"),
                },
            )
        yield event
