from pydantic import BaseModel, ConfigDict
from typing import Dict
import math
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
import re


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocab: Dict[str, int]
    id_to_token: Dict[int, str]
    functions: list[FunctionDefinition]

    def get_valid_tokens_for_name(
        self,
        generated_so_far: str
    ) -> list[int]:
        """Retourne les IDs de tokens valides pour le champ name."""
        valid_ids = []

        for func in self.functions:
            func_token_ids = self.model.encode(func.name)[0].tolist()

            if func.name.startswith(generated_so_far):
                already_encoded = (
                    self.model.encode(generated_so_far)[0].tolist()
                    if generated_so_far else []
                )
                pos = len(already_encoded)
                if pos < len(func_token_ids):
                    valid_ids.append(func_token_ids[pos])

        return list(set(valid_ids))

    def decode(self, prompt: str) -> dict:
        """Génère le JSON complet pour un prompt."""
        functions_desc = "\n".join([
            f"- {f.name}: {f.description}"
            for f in self.functions
        ])
        full_prompt = (
            f"Functions:\n{functions_desc}\n\n"
            f"Question: {prompt}\n\n"
            f"Answer in JSON: {{\"name\": \""
        )

        generated_name = ""

        while True:
            valid_ids = self.get_valid_tokens_for_name(generated_name)

            if not valid_ids:
                break

            current_text = full_prompt + generated_name
            input_ids = self.model.encode(current_text)[0].tolist()
            logits = self.model.get_logits_from_input_ids(input_ids)

            masked = [-math.inf] * len(logits)
            for token_id in valid_ids:
                masked[token_id] = logits[token_id]

            next_token_id = masked.index(max(masked))
            next_token_str = self.id_to_token[next_token_id]
            generated_name += next_token_str

            if any(f.name == generated_name for f in self.functions):
                break

        function = next(f for f in self.functions if f.name == generated_name)
        numbers = self.extract_numbers_from_prompt(prompt)
        strings = self.extract_strings_from_prompt(prompt)

        parameters = {}

        param_prompt = full_prompt + generated_name + '", "parameters": {'

        for param_name, param in function.parameters.items():
            if param.type == "number":
                value = self.generate_number(param_prompt, numbers)

                if value in numbers:
                    numbers.remove(value)

                parameters[param_name] = float(value) if value else 0.0

            elif param.type == "string":
                value = self.generate_string(param_prompt, strings)

                if value in strings:
                    strings.remove(value)

                parameters[param_name] = value

        return {
            "name": generated_name,
            "parameters": parameters
        }

    def extract_numbers_from_prompt(self, prompt: str) -> list[str]:
        """Extract numeric candidates from the prompt."""
        return re.findall(r'-?\d+\.?\d*', prompt)

    def extract_strings_from_prompt(self, prompt: str) -> list[str]:
        """Extract relevant word candidates from the prompt."""
        STOPWORDS = {
            "what", "is", "the", "a", "an", "of",
            "and", "greet", "reverse", "string",
            "replace", "all", "with", "function",
            "call", "using"
        }

        words = re.findall(r"[a-zA-Z]+", prompt)
        words = [w.lower() for w in words if len(w) > 1 and w.lower() not in STOPWORDS]

        # Heuristic: keep only last meaningful words (most likely arguments)
        if words:
            return words

        return []

    def get_valid_tokens_for_number(self, generated_so_far: str, candidates: list[str]) -> list[int]:
        valid_ids = []

        for number in candidates:
            if number.startswith(generated_so_far):
                number_token_ids = self.model.encode(number)[0].tolist()

                already_encoded = (
                    self.model.encode(generated_so_far)[0].tolist()
                    if generated_so_far else []
                )

                pos = len(already_encoded)

                if pos < len(number_token_ids):
                    valid_ids.append(number_token_ids[pos])

        return list(set(valid_ids))

    def get_valid_tokens_for_string(self, generated_so_far: str, candidates: list[str]) -> list[int]:
        valid_ids = []

        for word in candidates:
            if word.startswith(generated_so_far):
                word_token_ids = self.model.encode(word)[0].tolist()

                already_encoded = (
                    self.model.encode(generated_so_far)[0].tolist()
                    if generated_so_far else []
                )

                pos = len(already_encoded)

                if pos < len(word_token_ids):
                    valid_ids.append(word_token_ids[pos])

        return list(set(valid_ids))

    def generate_number(self, full_prompt: str, candidates: list[str]) -> str:
        generated_value = ""

        while True:
            valid_ids = self.get_valid_tokens_for_number(generated_value, candidates)

            if not valid_ids:
                break

            current_text = full_prompt + generated_value
            input_ids = self.model.encode(current_text)[0].tolist()
            logits = self.model.get_logits_from_input_ids(input_ids)

            masked = [-math.inf] * len(logits)
            for token_id in valid_ids:
                masked[token_id] = logits[token_id]

            next_token_id = masked.index(max(masked))
            next_token_str = self.id_to_token[next_token_id]

            generated_value += next_token_str

            if any(generated_value == n for n in candidates):
                break

        return generated_value

    def generate_string(self, full_prompt: str, candidates: list[str]) -> str:
        generated_value = ""

        while True:
            valid_ids = self.get_valid_tokens_for_string(generated_value, candidates)

            if not valid_ids:
                break

            current_text = full_prompt + generated_value
            input_ids = self.model.encode(current_text)[0].tolist()
            logits = self.model.get_logits_from_input_ids(input_ids)

            masked = [-math.inf] * len(logits)
            for token_id in valid_ids:
                masked[token_id] = logits[token_id]

            next_token_id = masked.index(max(masked))
            next_token_str = self.id_to_token[next_token_id]

            generated_value += next_token_str

            if any(generated_value == w for w in candidates):
                break

        return generated_value