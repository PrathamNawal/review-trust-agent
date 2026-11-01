import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.config_store import DEFAULT_CONFIG, DEFAULT_JEV_EVIDENCE_SOURCES
from dashboard.evidence import CARD_KEYS, build_time_aware_evidence, evidence_for_architecture
from dashboard.jev_agent import MODEL_ID
from dashboard.mechanism import JEV_SOURCES, build_dot, describe_mechanism
from scripts.tool_schemas import TOOL_DEFS
from scripts.tools import ToolBox


def _steps(mechanism):
    return {step["id"]: step for step in mechanism["steps"]}


class JevMechanismTests(unittest.TestCase):
    def test_source_list_cannot_drift_from_the_configurable_sources(self):
        self.assertEqual([key for key, *_ in JEV_SOURCES], list(DEFAULT_JEV_EVIDENCE_SOURCES))

    def test_default_config_describes_jev_choice_with_every_source_on(self):
        mech = describe_mechanism(DEFAULT_CONFIG)
        steps = _steps(mech)
        self.assertEqual(mech["architecture"], "jev")
        self.assertEqual(mech["name"], "Jev Choice")
        for key in DEFAULT_JEV_EVIDENCE_SOURCES:
            self.assertFalse(steps[f"src_{key}"]["off"])
        self.assertIn(MODEL_ID, dict(steps["jev"]["rows"])["Model"])
        self.assertFalse(steps["label"]["play"], "Choice has no cutoff to tune")

    def test_switching_a_source_off_dims_it_in_steps_and_diagram(self):
        config = {**DEFAULT_CONFIG, "jev_evidence_sources": ["reviewer_history", "business_trend"]}
        mech = describe_mechanism(config)
        steps = _steps(mech)
        self.assertFalse(steps["src_reviewer_history"]["off"])
        self.assertTrue(steps["src_business_text_similarity"]["off"])
        self.assertEqual(steps["src_business_text_similarity"]["tunables"], [("Include this source", "off")])
        self.assertIn("switched off", build_dot(mech))
        self.assertIn("2 of 4", mech["state_card"]["body"])

    def test_noul_method_exposes_the_cutoff_that_makes_the_label(self):
        config = {**DEFAULT_CONFIG, "jev_method": "noul", "jev_threshold": 0.62}
        mech = describe_mechanism(config)
        steps = _steps(mech)
        self.assertEqual(steps["label"]["tunables"], [("Cutoff", "0.62")])
        self.assertTrue(steps["label"]["play"])
        self.assertIn("0.62", mech["graph"]["nodes"]["label"]["label"])

    def test_routing_step_states_that_missing_confidence_goes_to_a_human(self):
        rule = dict(describe_mechanism(DEFAULT_CONFIG)["steps"][-2]["rows"])["Rule (compute_routing)"]
        self.assertIn("confidence missing", rule)
        self.assertIn("human queue", rule)

    def test_selected_step_is_drawn_heavier(self):
        mech = describe_mechanism(DEFAULT_CONFIG)
        selected_lines = [l for l in build_dot(mech, "jev").splitlines() if "penwidth=3.8" in l]
        self.assertEqual(len(selected_lines), 1)
        self.assertTrue(selected_lines[0].strip().startswith("jev ["))
        self.assertNotIn("penwidth=3.8", build_dot(mech))


class AgnoMechanismTests(unittest.TestCase):
    CONFIG = {**DEFAULT_CONFIG, "architecture": "agno", "provider": "groq",
              "model": "openai/gpt-oss-120b", "temperature": 0.2, "confidence_threshold": 0.9}

    def test_tools_and_terminal_step_come_from_the_real_agno_tools(self):
        mech = describe_mechanism(self.CONFIG)
        steps = _steps(mech)
        evidence_names = [d["name"] for d in TOOL_DEFS if d["name"] != "submit_judgment"]
        for name in evidence_names:
            self.assertIn(f"t_{name}", steps)
        self.assertEqual(steps["a_term"]["title"], "submit_judgment")

    def test_threshold_and_memory_facts_are_read_not_typed(self):
        mech = describe_mechanism(self.CONFIG)
        self.assertEqual(_steps(mech)["a_router"]["tunables"], [("Confidence threshold", "0.90")])
        self.assertIn("db=None", mech["memory_card"]["source"])
        self.assertEqual(_steps(mech)["a_model"]["tunables"][1:], [
            ("Provider", "groq"), ("Model", "openai/gpt-oss-120b"), ("Temperature", "0.20")])

    def test_unknown_architecture_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            describe_mechanism({**DEFAULT_CONFIG, "architecture": "something-new"})


class EvidenceParityTests(unittest.TestCase):
    """People must be shown the evidence the active architecture actually received."""

    def setUp(self):
        rows = [  # review_id, user, prod, date
            (0, 1, 10, "2012-01-01"),   # earlier than the case
            (1, 1, 10, "2012-03-01"),   # the case under investigation
            (2, 1, 10, "2012-06-01"),   # LATER than the case
            (3, 1, 10, "2012-03-01"),   # same day as the case
        ]
        df = pd.DataFrame({
            "user_id": [r[1] for r in rows], "prod_id": [r[2] for r in rows],
            "rating": [4.0, 5.0, 5.0, 5.0], "date": pd.to_datetime([r[3] for r in rows]),
            "review_text": ["decent pasta", "great pasta here", "great pasta here again", "great pasta"],
            "filtered": [False, False, True, False],
        }, index=pd.Index([r[0] for r in rows], name="review_id"))
        by_user = {1: df.sort_values("date").index.tolist()}
        by_prod = {10: df.sort_values("date").index.tolist()}
        self.toolbox = ToolBox({"df": df, "by_user": by_user, "by_prod": by_prod})
        self.review = {"review_id": 1, "user_id": "1", "prod_id": "10", "date": "2012-03-01",
                       "review_text": "great pasta here", "rating": 5.0}

    def test_time_aware_evidence_excludes_later_and_same_day_reviews(self):
        evidence = build_time_aware_evidence(self.toolbox, self.review, ["reviewer_history", "business_trend"])
        self.assertEqual(evidence["reviewer_history"]["other_review_count"], 1)
        self.assertEqual(evidence["business_trend"]["other_review_count"], 1)

    def test_jev_cards_use_time_aware_evidence_and_agno_cards_use_the_original_lookups(self):
        jev_cards, jev_note = evidence_for_architecture(self.toolbox, self.review, "jev")
        agno_cards, agno_note = evidence_for_architecture(self.toolbox, self.review, "agno")
        self.assertEqual([k for k, _ in jev_cards], list(CARD_KEYS.values()))
        self.assertEqual(dict(jev_cards)["reviewer_history_lookup"]["other_review_count"], 1)
        self.assertEqual(dict(agno_cards)["reviewer_history_lookup"]["other_review_count"], 3)
        self.assertIn("before this one", jev_note)
        self.assertIn("later reviews", agno_note)


if __name__ == "__main__":
    unittest.main()
