from afnio.models.openai import AsyncOpenAI, OpenAI

# Only models listed in this registry can be saved or loaded as part of the
# optimizer and module state. Ensure any new model types are registered here
# to enable proper serialization and deserialization.
MODEL_REGISTRY = {
    "OpenAI": OpenAI,
    "AsyncOpenAI": AsyncOpenAI,
}
