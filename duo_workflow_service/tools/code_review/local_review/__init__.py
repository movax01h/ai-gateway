"""Deterministic acquisition of a local review target, for the `checkout` surface.

`transport` is the executor boundary: every git command and file read this package makes goes
through it, and it is what makes the package read-only and what stops configured git filters
from executing.
"""
