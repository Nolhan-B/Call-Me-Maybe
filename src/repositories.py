import json
from typing import List
from pydantic import BaseModel, ValidationError
from src.models import FunctionDefinition, Prompt


class FunctionRepository(BaseModel):
    """Repository for function definitions."""
    filepath: str

    def get_all(self) -> List[FunctionDefinition | None]:
        """Load and return all function definitions."""
        try:
            with open(self.filepath, "r") as file:
                raw = json.load(file)
            return [FunctionDefinition(**item) for item in raw]
        except FileNotFoundError:
            print(f"Error: file not found ({self.filepath})")
            return []
        except json.decoder.JSONDecodeError:
            print(f"Error: json is not valid ({self.filepath})")
            return []
        except ValidationError as e:
            print(f"Error: invalid structure: {e}")
            return []

class PromptRepository(BaseModel):
    """Repository for prompt definitions."""

    filepath: str

    def get_all(self) -> List[Prompt | None]:
        """Load and return all function definitions."""
        try:
            with open(self.filepath, "r") as file:
                raw = json.load(file)
            return [Prompt(**item) for item in raw]
        except FileNotFoundError:
            print(f"Error: file not found ({self.filepath})")
            return []
        except json.decoder.JSONDecodeError:
            print(f"Error: json is not valid ({self.filepath})")
            return []
        except ValidationError as e:
            print(f"Error: invalid structure: {e}")
            return []