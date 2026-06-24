from typing import Literal

ScanTypeLiteral = Literal["FULL", "INCREMENTAL"]
AscpSeverityLiteral = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

__all__ = ["AscpSeverityLiteral", "ScanTypeLiteral"]
