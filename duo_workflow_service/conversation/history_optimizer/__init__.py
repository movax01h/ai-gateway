"""History optimizer abstraction.

The framework lives at the top level: ``base`` (abstract class),
``pipeline`` (ordered composition), ``builder`` (``FlowContext`` +
construction from typed feature flags), and ``schema`` (config + result
types).

The ``optimizers/`` subpackage holds concrete implementations
(``CompactionOptimizer``, ``LegacyTrimOptimizer``). To add a new optimizer,
drop a module in there and wire it into ``builder``.

This package is the canonical home for ``CompactionConfig`` and
``CompactionResult``; the legacy
``duo_workflow_service.conversation.compaction`` package re-exports them
for backwards compatibility while callers migrate.
"""
