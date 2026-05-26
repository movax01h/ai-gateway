from pydantic import BaseModel

EMBEDDING_MODEL_NAME = "embedding"


class EmbeddingsRequest(BaseModel):
    class ModelMetadata(BaseModel):
        provider: str
        identifier: str
        name: str | None = None
        endpoint: str | None = None
        api_key: str | None = None

    model_metadata: ModelMetadata
    contents: list[str]
    dimensions: int | None = None
    litellm_drop_params: bool | None = None


class EmbeddingsResponse(BaseModel):
    class Prediction(BaseModel):
        embedding: list[float]
        index: int

    class Model(BaseModel):
        engine: str
        name: str
        identifier: str

    model: Model
    predictions: list[Prediction]
