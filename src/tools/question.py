"""Interactive question tool — allows the agent to ask the user questions.

Uses LangGraph interrupt() for HITL pause.
The UI displays the question and captures the user's response.
"""

import json
from langchain_core.tools import tool


@tool
def question(
    question_text: str,
    options: str = "",
    allow_custom: bool = True,
) -> str:
    """Ask the user a question and get their response.
    
    The agent pauses and waits for the user to respond.
    Use this when you need clarification, confirmation, or user input
    before proceeding with a task.

    Args:
        question_text: The question to ask the user
        options: Optional JSON array of predefined options, e.g. '["yes", "no"]'.
                 Leave empty for free-text input.
        allow_custom: Whether the user can provide a custom answer beyond options

    Returns:
        The user's response or selected option
    """
    # This tool triggers a LangGraph interrupt via deepagents HITL.
    # The actual pause/resume is handled by interrupt_before=["tools"].
    # The tool input is shown to the user; the return value is the user's response.
    parsed_options = None
    if options.strip():
        try:
            parsed_options = json.loads(options)
        except json.JSONDecodeError:
            parsed_options = options  # pass through as raw string

    return json.dumps({
        "tool": "question",
        "question": question_text,
        "options": parsed_options,
        "allow_custom": allow_custom,
    })


def create_question_tool():
    return question
