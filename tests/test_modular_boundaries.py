import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import agent_service, config_store
from dashboard.agent_architectures import register_architecture, run_investigation_stream
from dashboard.decision_policy import compute_routing
from dashboard.metrics import binary_classification_metrics, wilson_interval
from dashboard.evaluation import compare_configurations, evaluate_reviews
from dashboard.providers import PROVIDER_PRESETS
from scripts.tool_schemas import SYSTEM_PROMPT, TOOL_DEFS, dispatch_tool_call
from dashboard.live_agent import _build_tools


class RoutingPolicyTests(unittest.TestCase):
    def test_confident_fake_is_auto_removed(self):
        self.assertEqual(compute_routing(True, 0.75, 0.75), "auto_remove")

    def test_confident_genuine_is_ignored(self):
        self.assertEqual(compute_routing(False, 0.76, 0.75), "ignore")

    def test_low_confidence_always_queues(self):
        self.assertEqual(compute_routing(True, 0.74, 0.75), "queue")
        self.assertEqual(compute_routing(False, 0.20, 0.75), "queue")


class MetricsTests(unittest.TestCase):
    def test_fraud_positive_metrics_expose_missed_fraud(self):
        metrics = binary_classification_metrics(
            [True, True, False, False], [False, False, True, False]
        )
        self.assertEqual(metrics["false_negative"], 2)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.25)

    def test_undefined_metrics_remain_undefined(self):
        metrics = binary_classification_metrics([False, False], [False, False])
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["f1"])

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(0, 6)
        self.assertLessEqual(low, 0)
        self.assertGreater(high, 0)


class ArchitectureRegistryTests(unittest.TestCase):
    def test_registered_architecture_emits_shared_trace(self):
        register_architecture("test-runner", lambda review, config: iter([{"type": "turn", "turn": 0}]))
        events = list(run_investigation_stream({}, {"architecture": "test-runner"}))
        self.assertEqual(events, [{"type": "turn", "turn": 0}])

    def test_unknown_architecture_is_a_trace_error(self):
        events = list(run_investigation_stream({}, {"architecture": "missing-runner"}))
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("Unknown agent architecture", events[0]["message"])


class ProviderRegistryTests(unittest.TestCase):
    def test_current_providers_expose_api_key_and_lazy_model_location(self):
        self.assertEqual(set(PROVIDER_PRESETS), {"groq", "openrouter", "gemini", "claude"})
        for spec in PROVIDER_PRESETS.values():
            self.assertIn("api_key_env", spec)
            self.assertIn("module", spec)
            self.assertIn("class_name", spec)


class EvidenceToolRegistryTests(unittest.TestCase):
    def test_registry_dispatch_preserves_self_exclusion_only_for_same_reviewer(self):
        toolbox = Mock()
        review = {"review_id": 7, "user_id": "u1", "prod_id": "p1"}
        dispatch_tool_call(toolbox, review, "reviewer_history_lookup", {"user_id": "u1"})
        toolbox.reviewer_history_lookup.assert_called_once_with("u1", exclude_review_id=7)

        dispatch_tool_call(toolbox, review, "reviewer_history_lookup", {"user_id": "u2"})
        self.assertEqual(toolbox.reviewer_history_lookup.call_args.args, ("u2",))
        self.assertEqual(toolbox.reviewer_history_lookup.call_args.kwargs, {"exclude_review_id": None})

    def test_agno_adapter_uses_canonical_tool_names_and_descriptions(self):
        toolbox = Mock()
        tools = _build_tools(toolbox, {"review_id": 1, "user_id": "u1", "prod_id": "p1"})
        definitions = {item["name"]: item for item in TOOL_DEFS}
        self.assertEqual([item.name for item in tools], [item["name"] for item in TOOL_DEFS])
        for adapter in tools:
            self.assertEqual(adapter.description, definitions[adapter.name]["description"])
            self.assertEqual(adapter.parameters, definitions[adapter.name]["parameters"])
        tools[0].entrypoint(user_id="u1")
        toolbox.reviewer_history_lookup.assert_called_once_with("u1", exclude_review_id=1)
        self.assertEqual(tools[-1].entrypoint(predicted_filtered=True), "submitted")


class InvestigationServiceTests(unittest.TestCase):
    def test_completed_verdict_is_logged_once_with_shared_metadata(self):
        review = {"review_id": 42}
        config = {
            "version": "v3", "architecture": "agno", "provider": "groq",
            "model": "example-model", "temperature": 0.2,
        }
        tracker = Mock()
        trace = [
            {"type": "tool_call", "tool": "reviewer_history_lookup"},
            {"type": "final", "predicted_filtered": True, "confidence": 0.9,
             "primary_signal": "reviewer_pattern", "reasoning": "Evidence supports filtering.",
             "turns_used": 2},
        ]
        with patch.object(agent_service, "run_investigation_stream", return_value=iter(trace)):
            self.assertEqual(list(agent_service.investigate_and_log(review, config, True, tracker)), trace)

        tracker.log_judgment.assert_called_once_with(
            judge_type="agent", config_version="v3", review_id=42, true_filtered=True,
            predicted_filtered=True, confidence=0.9, primary_signal="reviewer_pattern",
            reasoning="Evidence supports filtering.", turns_used=2, n_tool_calls=1,
            provider="groq", model="example-model", temperature=0.2,
            extra_tags={"framework": "agno", "architecture": "agno"},
        )

    def test_incomplete_or_error_run_is_not_logged(self):
        tracker = Mock()
        events = [
            {"type": "error", "message": "provider error"},
            {"type": "final", "_incomplete": True, "reasoning": "turn cap"},
        ]
        with patch.object(agent_service, "run_investigation_stream", return_value=iter(events)):
            list(agent_service.investigate_and_log({}, {"version": "v0"}, False, tracker))
        tracker.log_judgment.assert_not_called()


class ConfigCompatibilityTests(unittest.TestCase):
    def test_default_prompt_has_one_source_of_truth(self):
        self.assertEqual(config_store.DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPT)

    def test_deploy_persists_architecture_and_legacy_config_defaults_to_agno(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "agent_configs.json"
            with patch.object(config_store, "STORE_PATH", str(store_path)):
                config_store.deploy_new_version(
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.2,
                    provider="groq",
                    model="openai/gpt-oss-120b",
                    confidence_threshold=0.75,
                )
                second = config_store.deploy_new_version(
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.3,
                    provider="groq",
                    model="openai/gpt-oss-120b",
                    confidence_threshold=0.75,
                    architecture="test-runner",
                )
                self.assertEqual(second["architecture"], "test-runner")
                self.assertEqual(config_store.get_active_config()["architecture"], "test-runner")
                self.assertEqual(config_store.list_versions()[0]["architecture"], "agno")


class EvaluationTests(unittest.TestCase):
    def test_fixed_review_evaluation_uses_injected_runner(self):
        reviews = [{"review_id": 1, "filtered": True}, {"review_id": 2, "filtered": False}]

        def runner(review, _config):
            yield {"type": "tool_call", "tool": "evidence"}
            yield {"type": "final", "predicted_filtered": review["filtered"]}

        rows, metrics = evaluate_reviews(reviews, {"architecture": "fake"}, runner=runner)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["completed"], 2)
        self.assertEqual(rows["tool_calls"].tolist(), [1, 1])

    def test_errors_are_not_counted_as_completed_evaluations(self):
        review = {"review_id": 3, "filtered": True}
        rows, metrics = evaluate_reviews(
            [review], {"architecture": "fake"},
            runner=lambda *_: iter([{"type": "error", "message": "offline"}]),
        )
        self.assertEqual(metrics["completed"], 0)
        self.assertIsNone(metrics["recall"])
        self.assertEqual(rows.iloc[0]["error"], "offline")

    def test_configuration_comparison_uses_same_reviews_and_reports_recall_delta(self):
        reviews = [{"review_id": 9, "filtered": True}, {"review_id": 10, "filtered": False}]

        def runner(_review, config):
            yield {"type": "final", "predicted_filtered": config["predict_fraud"]}

        result = compare_configurations(
            {"predict_fraud": False}, {"predict_fraud": True}, reviews, runner=runner,
        )
        self.assertEqual(result["review_ids"], [9, 10])
        self.assertEqual(result["delta"]["recall"], 1.0)
        self.assertEqual(result["baseline"]["metrics"]["false_negative"], 1)
        self.assertEqual(result["candidate"]["metrics"]["false_positive"], 1)


if __name__ == "__main__":
    unittest.main()
