import asyncio
from typing import Any, Dict

import structlog

from contract import contract_pb2


async def _execute_action(metadata: Dict[str, Any], action: contract_pb2.Action):
    outbox: asyncio.Queue = metadata["outbox"]
    inbox: asyncio.Queue = metadata["inbox"]
    log = structlog.stdlib.get_logger("workflow")

    if action.runCommand:
        log.debug(
            "Attempting action from the egress queue",
            requestID=action.requestID,
            action_class=contract_pb2.RunCommandAction,
        )
    elif action.runHTTPRequest:
        log.debug(
            "Attempting action from the egress queue",
            requestID=action.requestID,
            action_class=contract_pb2.RunHTTPRequest,
        )
    elif action.runReadFile:
        log.debug(
            "Attempting action from the egress queue",
            requestID=action.requestID,
            action_class=contract_pb2.ReadFile,
        )
    elif action.runWriteFile:
        log.debug(
            "Attempting action from the egress queue",
            requestID=action.requestID,
            action_class=contract_pb2.WriteFile,
        )
    elif action.runGitCommand:
        log.debug(
            "Attempting action from the egress queue",
            requestID=action.requestID,
            action_class=contract_pb2.RunGitCommand,
        )
    elif action.runEditFile:
        log.debug(
            "Attempting action from the egress queue",
            requestID=action.requestID,
            action_class=contract_pb2.EditFile,
        )

    await outbox.put(action)

    event: contract_pb2.ClientEvent = await inbox.get()

    if event.actionResponse:
        log.debug(
            "Read ClientEvent into the ingres queue",
            requestID=event.actionResponse.requestID,
        )

    inbox.task_done()
    return event.actionResponse.response
