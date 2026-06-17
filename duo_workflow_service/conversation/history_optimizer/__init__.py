"""History optimizer abstraction.

The framework lives at the top level: ``base`` (abstract class),
``pipeline`` (ordered composition), ``builder`` (``FlowContext`` +
construction from configs), ``schema`` (config + result types), and
``validation`` (config-list rules).

The ``optimizers/`` subpackage holds concrete implementations
(``CompactionOptimizer``, ``LegacyTrimOptimizer``). To add a new optimizer,
drop a module in there and wire its config into ``schema`` and ``builder``.

This package is the canonical home for ``CompactionConfig`` and
``CompactionResult``; the legacy
``duo_workflow_service.conversation.compaction`` package re-exports them
for backwards compatibility while callers migrate.
"""
