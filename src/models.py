from pydantic import BaseModel
from typing import Dict


class ParameterType(BaseModel):
    """Represents the type of a single
    function parameter or return value.

    Attributes:
        type: The type name, e.g. 'number', 'string', 'boolean'.
    """
    type: str


class FunctionDefinition(BaseModel):
    """Represents a callable function with its metadata
    and parameter schema.

    Attributes:
        name: The function identifier, e.g. 'fn_add_numbers'.
        description: Human-readable description of what the function does.
        parameters: Mapping of parameter names to their types.
        returns: The return type of the function.
    """
    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType


class Prompt(BaseModel):
    """Represents a single natural language prompt to process.

    Attributes:
        prompt: The raw natural language input string.
    """
    prompt: str
