"""Performance dashboard view backed by the shared MLflow metrics service."""

import streamlit as st


def render_performance(mlflow_tracking):
    st.title("Track Performance")
    st.caption("Every human and agent judgment, grouped by config version — logged to MLflow, not a hand-rolled CSV.")

    summary = mlflow_tracking.summarize_by_group()
    if summary.empty:
        st.info("No judgments logged yet. Go try it as a human or the agent first.")
    else:
        agent_summary = summary[summary["tags.judge_type"] == "agent"].copy()
        if len(agent_summary) >= 2:
            agent_summary["_order"] = range(len(agent_summary))
            prev_row, latest_row = agent_summary.iloc[-2], agent_summary.iloc[-1]
            delta = latest_row["accuracy"] - prev_row["accuracy"]
            arrow = "📈" if delta > 0 else ("📉" if delta < 0 else "➡️")
            st.markdown(
                f"### {arrow} Your last agent config change moved accuracy from "
                f"**{prev_row['accuracy']:.0%}** (`{prev_row['tags.config_version']}`) to "
                f"**{latest_row['accuracy']:.0%}** (`{latest_row['tags.config_version']}`)"
            )
        else:
            st.markdown("Deploy at least two config versions and run the agent under each to see a before/after delta here.")

        st.divider()
        st.subheader("Performance by judge type & config version")
        display_cols = [
            "tags.judge_type", "tags.config_version", "n", "recall", "precision", "f1",
            "balanced_accuracy", "accuracy", "accuracy_ci", "specificity", "false_positive_rate",
            "true_positive", "false_positive", "false_negative", "true_negative",
        ]
        st.dataframe(summary[[c for c in display_cols if c in summary.columns]], width="stretch")

        st.subheader("Fraud recall by config version (agent only)")
        if not agent_summary.empty:
            st.bar_chart(
                agent_summary.set_index("tags.config_version")[["recall"]],
                width="stretch",
            )

        st.subheader("Accuracy by config version")
        st.bar_chart(
            summary.pivot_table(index="tags.config_version", columns="tags.judge_type", values="accuracy"),
            width="stretch",
        )

        st.subheader("Precision / Recall by config version (agent only)")
        if not agent_summary.empty:
            pr_pivot = agent_summary.set_index("tags.config_version")[["precision", "recall"]]
            st.bar_chart(pr_pivot, width="stretch")

        with st.expander("Raw MLflow run log"):
            st.dataframe(mlflow_tracking.get_all_runs(), width="stretch")
