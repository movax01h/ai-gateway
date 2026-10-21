"""The typed image result surfaced from ``ActionResponse.imageResponse``.

Kept a leaf module: ``executor/action.py`` and ``entities/image_response.py`` both import
this type, and anything heavier here would recreate the action -> tools -> action cycle.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageActionResult:
    """A typed image result (``ActionResponse.imageResponse``) from the executor.

    ``data`` is the encoded image file as the client read it. Validating format, size and
    signature belongs to the consumer, ``entities/image_response.py``.
    """

    mime_type: str
    data: bytes
