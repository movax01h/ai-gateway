from typing import Literal

ScanTypeLiteral = Literal["FULL", "INCREMENTAL"]
AscpSeverityLiteral = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
AscpSecurityBoundaryLiteral = Literal[
    "NETWORK_ACCESS",
    "USER_INPUT",
    "PARTNER_BOUNDARY",
    "TRUSTED_SERVICE",
    "INTERNAL_ONLY",
    "ISOLATED",
]

__all__ = [
    "AscpSecurityBoundaryLiteral",
    "AscpSeverityLiteral",
    "ScanTypeLiteral",
]
