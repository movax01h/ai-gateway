from typing import Any, Optional, Protocol, TypeAlias

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableBinding

# NOTE: Do not change this to `BaseChatModel | RunnableBinding`. You'd think that's just equivalent, right? WRONG. If
# you do that, you'll get `object has no attribute 'get'` when you use a `RummableBinding`. Why? I have no idea.
# https://docs.python.org/3/library/stdtypes.html#types-union makes no mention of the order mattering. This might be
# a bug with Pydantic's type validations
Model: TypeAlias = RunnableBinding | BaseChatModel


class TypeModelFactory(Protocol):
    def __call__(self, *, model: str, **kwargs: Optional[Any]) -> Model: ...
