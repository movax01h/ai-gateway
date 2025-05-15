from typing import Optional

from pydantic import BaseModel


class ParsedSlashCommand(BaseModel):
    """
    Represents a slash command that has been parsed from the user's input goal.
    Contains the extracted command type and any additional text provided.
    """

    command_type: str = ""
    remaining_text: Optional[str] = None


class SlashCommandsGoalParser:
    """
    Parser for user input containing slash commands.

    This class is responsible for extracting the command type and
    separating it from any additional text in the goal.
    """

    def parse(self, goal: str) -> ParsedSlashCommand:
        """
        Parse a goal string to extract the slash command and remaining text.

        Args:
            goal: The user input string (e.g., "/explain This code is confusing")

        Returns:
            ParsedSlashCommand containing the command type and remaining text
        """
        command_type = ""
        remaining_text = ""

        # Separate slash command and the rest of the string in user's input.

        return ParsedSlashCommand(
            command_type=command_type, remaining_text=remaining_text
        )
