"""Reusable Streamlit presentation components for the dashboard."""

import streamlit as st

TOOL_TITLES = {
    "reviewer_history_lookup": "🕵️ Reviewer History",
    "business_trend_lookup": "📈 Business Trend",
    "text_similarity_reviewer": "🔎 Text Similarity vs. Reviewer",
    "text_similarity_business": "🔎 Text Similarity vs. Business",
}

_CSS = """
<style>
:root {
    --rta-danger-bg: #fde8e8; --rta-danger-fg: #b91c1c;
    --rta-warning-bg: #fef3c7; --rta-warning-fg: #92400e;
    --rta-safe-bg: #d1fae5; --rta-safe-fg: #065f46;
}
.rta-hero-stat { padding: 4px 0; }
.rta-hero-stat-label { font-size: 0.8rem; color: var(--rta-hero-muted, #6b7280); font-weight: 500; }
.rta-hero-stat-value { font-size: 2.1rem; font-weight: 700; line-height: 1.3; }
.rta-hero-stat-caption { font-size: 0.78rem; color: var(--rta-hero-muted, #6b7280); margin-top: 2px; }
.rta-badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-weight: 600; font-size: 0.85rem; white-space: nowrap;
}
.rta-badge-danger { background: var(--rta-danger-bg); color: var(--rta-danger-fg); }
.rta-badge-warning { background: var(--rta-warning-bg); color: var(--rta-warning-fg); }
.rta-badge-safe { background: var(--rta-safe-bg); color: var(--rta-safe-fg); }
.rta-card-title {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--rta-hero-muted, #6b7280); font-weight: 600; margin-bottom: 6px;
}
.rta-snippet {
    font-size: 0.85rem; font-style: italic; color: var(--rta-hero-muted, #6b7280);
    border-left: 2px solid #d1d5db; padding-left: 8px; margin: 4px 0;
}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def render_badge(text, tone):
    st.markdown(f'<span class="rta-badge rta-badge-{tone}">{text}</span>', unsafe_allow_html=True)


def render_hero_stat(container, label, value, caption, tone="neutral"):
    color = {"danger": "var(--rta-danger-fg)", "warning": "var(--rta-warning-fg)", "neutral": "inherit"}[tone]
    container.markdown(
        f'<div class="rta-hero-stat">'
        f'<div class="rta-hero-stat-label">{label}</div>'
        f'<div class="rta-hero-stat-value" style="color:{color}">{value}</div>'
        f'<div class="rta-hero-stat-caption">{caption}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_evidence_card(container, tool_key, data):
    """Render real evidence output as a compact metric card."""
    with container.container(border=True):
        st.markdown(f'<div class="rta-card-title">{TOOL_TITLES.get(tool_key, tool_key)}</div>',
                    unsafe_allow_html=True)
        if data.get("note"):
            st.caption(f"ℹ️ {data['note']}")

        skip_keys = {"note", "similar_snippets", "user_id", "prod_id"}
        metric_pairs = [(k.replace("_", " ").capitalize(), v) for k, v in data.items() if k not in skip_keys]
        if metric_pairs:
            cols = st.columns(min(len(metric_pairs), 3) or 1)
            for i, (label, val) in enumerate(metric_pairs):
                cols[i % len(cols)].metric(label, "—" if val is None else val)

        for snippet in (data.get("similar_snippets") or [])[:2]:
            st.markdown(f'<div class="rta-snippet">"{snippet["snippet"][:130]}…" '
                        f'(similarity {snippet["similarity"]:.2f})</div>', unsafe_allow_html=True)


def resolve_tool_key(tool_name, tool_input):
    if tool_name == "text_similarity_check":
        return f"text_similarity_{tool_input.get('compare_against', 'reviewer')}"
    return tool_name


def render_evidence_grid(container, toolbox, review_id, user_id, prod_id, review_text, date):
    """Render the same four evidence lookups used by the agent."""
    first, second = container.columns(2)
    render_evidence_card(first, "reviewer_history_lookup",
                         toolbox.reviewer_history_lookup(user_id, exclude_review_id=review_id))
    render_evidence_card(second, "business_trend_lookup",
                         toolbox.business_trend_lookup(prod_id, date, exclude_review_id=review_id))
    third, fourth = container.columns(2)
    render_evidence_card(third, "text_similarity_reviewer",
                         toolbox.text_similarity_check(review_text, "reviewer", user_id,
                                                       exclude_review_id=review_id))
    render_evidence_card(fourth, "text_similarity_business",
                         toolbox.text_similarity_check(review_text, "business", prod_id,
                                                       exclude_review_id=review_id))
