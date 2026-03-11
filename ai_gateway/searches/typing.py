from pydantic import BaseModel


class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict
