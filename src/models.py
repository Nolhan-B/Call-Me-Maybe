from pydantic import BaseModel
from typing import Dict

class ParameterType(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType

class Prompt(BaseModel):
    prompt: str