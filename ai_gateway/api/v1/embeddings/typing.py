from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    class ModelMetadata(BaseModel):
        provider: str
        identifier: str

    model_metadata: ModelMetadata
    contents: list[str]


class EmbeddingsResponse(BaseModel):
    class Prediction(BaseModel):
        embedding: list[float]
        index: int

    class Model(BaseModel):
        engine: str
        name: str

    model: Model
    predictions: list[Prediction]
