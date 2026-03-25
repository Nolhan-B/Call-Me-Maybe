from pydantic import BaseModel, Field


class ConfigModel(BaseModel):
    """Configuration model holding file paths for the program.

    Attributes:
        fct_file: Path to the function definitions JSON file.
        input_file: Path to the input prompts JSON file.
        output_file: Path to the output results JSON file.
    """
    fct_file: str = Field(default="data/input/functions_definition.json")
    input_file: str = Field(default="data/input/function_calling_tests.json")
    output_file: str = Field(default="data/output/function_calls.json")
