from pydantic import BaseModel, Field

class ConfigModel(BaseModel):
	fct_file: str = Field(default="data/input/functions_definition.json")
	input_file: str = Field(default="data/input/function_calling_tests.json")
	output_file: str = Field(default="data/output/function_calls.json")

