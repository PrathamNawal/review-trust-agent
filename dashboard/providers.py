"""Provider adapters for the Agno-backed agent.

Adding a provider should require one registry entry here and its SDK/model
dependency, without changing investigation orchestration or the dashboard.
Model classes are imported on demand so an unavailable optional provider SDK
doesn't prevent the rest of the application from starting.
"""

from importlib import import_module


PROVIDER_PRESETS = {
    "groq": {"api_key_env": "GROQ_API_KEY", "module": "agno.models.groq", "class_name": "Groq"},
    "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "module": "agno.models.openrouter", "class_name": "OpenRouter"},
    "gemini": {"api_key_env": "GOOGLE_API_KEY", "module": "agno.models.google", "class_name": "Gemini"},
    "claude": {"api_key_env": "ANTHROPIC_API_KEY", "module": "agno.models.anthropic", "class_name": "Claude"},
}


def build_model(provider, model_id, api_key, temperature):
    """Construct a provider model from the common model interface."""
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ValueError(f"Unknown provider: {provider}")
    model_cls = getattr(import_module(preset["module"]), preset["class_name"])
    return model_cls(id=model_id, api_key=api_key, temperature=temperature)
