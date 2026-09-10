"""The typed image result surfaced from ``ActionResponse.imageResponse``.

A leaf module on purpose: both the executor seam (``executor/action.py``) and
the consumer (``entities/image_response.py``) import this type, and anything
heavier here would recreate the action -> tools -> action import cycle.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageActionResult:
    """A typed image result (``ActionResponse.imageResponse``) from the executor.

    ``data`` is the encoded image file bytes exactly as the client read them;
    validation (allowed formats, size cap, signature) belongs to the consumer,
    see ``entities/image_response.py``.
    """

    mime_type: str
    data: bytes
