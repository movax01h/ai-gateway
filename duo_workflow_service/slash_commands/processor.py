import structlog

from duo_workflow_service.slash_commands.definition import SlashCommandDefinition
from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandConfigError,
    SlashCommandError,
    log_command_error,
)
from duo_workflow_service.slash_commands.goal_parser import parse
from lib.result import Error, Ok, Result

log = structlog.stdlib.get_logger("slash_commands")


class SlashCommandsProcessor:
    """Class for processing slash commands.

    This class encapsulates the logic for processing slash commands, handling their parameters, and generating
    appropriate responses.
    """

    def process(self, message: str) -> Result:
        """Process a slash command.

        Args:
            message: The message text to process
            context_element_type: The type of context element to be used in the prompt

        Returns:
            Result containing SlashCommandResult if successful, or Exception if an error occurred
        """

        try:
            command_name, remaining_text = parse(message)

            if not command_name or not message.strip().startswith("/"):
                return Error("The message does not contain a command after the slash.")

            try:
                command_definition = (
                    SlashCommandDefinition.load_slash_command_definition(command_name)
                )
            except SlashCommandConfigError as e:
                log_command_error(command_name, e)
                return Error(e)

            # Replace the <ContextElementType> with the actual context element type variable
            goal = command_definition.goal

            # Build the result dictionary
            slash_command_result = {
                "success": True,
                "goal": goal,
                "parameters": command_definition.parameters,
                "message_context": remaining_text,
                "error": None,
                "command_name": command_name,
            }

            return Ok(slash_command_result)

        except SlashCommandError as e:
            log_command_error(command_name=None, error=e)
            return Error(e)
        except Exception as e:
            log.error(f"Error processing slash command: {str(e)}")
            return Error(e)
