from pydantic import BaseModel, ConfigDict
from typing import Dict, Any
import math
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
import re


class ConstrainedDecoder(BaseModel):
    """Decode natural language prompts into structured function call dicts.

    Uses constrained decoding to guarantee valid outputs by masking invalid
    tokens to -inf at each generation step, forcing the LLM to only produce
    tokens that match the expected schema.

    Attributes:
        model: The small LLM used for token generation.
        vocab: Mapping from token string to token ID.
        id_to_token: Reverse mapping from token ID to token string.
        functions: List of available function definitions to select from.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocab: Dict[str, int]
    id_to_token: Dict[int, str]
    functions: list[FunctionDefinition]

    def get_valid_tokens_for_name(
            self,
            generated_so_far: str
    ) -> list[int]:
        """Return valid token IDs for the next function name token.

        Computes which tokens can be generated next without invalidating
        all function name candidates. A token is valid if it matches the
        next token in at least one function name that starts with the
        string generated so far.

        Args:
            generated_so_far: The function name prefix generated so far.

        Returns:
            List of token IDs that would continue at least one valid
            function name.
        """

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

    def decode(self, prompt: str) -> Dict[str, Any]:
        """Generate a complete function call dict from
        a natural language prompt.

        Selects the most appropriate function using
        constrained decoding over function names,
        then extracts and generates each parameter value using
        candidates extracted from the prompt text.

        Args:
            prompt: The natural language prompt to process.

        Returns:
            A dict with keys 'prompt', 'name', and 'parameters'. I
            f no function matched, 'name' is an empty string
            and 'parameters' is an empty dict.
        """

        functions_desc = "\n".join([
            f"- {f.name}: {f.description}"
            for f in self.functions
        ])
        full_prompt = (
            "You are a function-calling assistant.\n"
            "Given a user request, identify the correct function and extract ONLY "
            "the argument values from the request.\n"
            "Do NOT copy the full sentence as a value. Extract only the relevant part.\n"
            f"You are a function selector.\n\n"
            f"Available functions:\n{functions_desc}\n\n"
            f"Examples:\n"
            f"Question: What is the sum of 1 and 2?\n"
            f"Function: fn_add_numbers\n\n"
            f"Question: Replace all digits in 'abc 12' with X\n"
            f"Function: fn_substitute_string_with_regex\n\n"
            f"Question: Replace all vowels in 'hello' with *\n"
            f"Function: fn_substitute_string_with_regex\n\n"
            f"Question: {prompt}\n"
            f"Function: \""
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

        function = next(
            (f for f in self.functions if f.name == generated_name), None
        )
        if function is None:
            print(f"Error: no function matched '{generated_name}'")
            return {"prompt": prompt, "name": "", "parameters": {}}

        numbers = self.extract_numbers_from_prompt(prompt)
        quoted_strings = self.extract_quoted_strings(prompt)
        word_strings = self.extract_word_strings(prompt)

        parameters: Dict[str, Any] = {}
        param_prompt = full_prompt + generated_name + '", "parameters": {'

        for param_name, param in function.parameters.items():
            crt_param_prompt = param_prompt + f'"{param_name}": "'
            if param.type in ("number", "integer"):
                value = self.generate_number(
                    param_prompt + f'"{param_name}": ', numbers
                )
                print(f"DEBUG generate_number({param_name}) = '{value}', candidates={numbers}")
                if value in numbers:
                    numbers.remove(value)
                if param.type == "number":
                    parameters[param_name] = float(value) if value else 0.0
                if param.type == "integer":
                    parameters[param_name] = int(value) if value else 0
                param_prompt += f'"{param_name}": {parameters[param_name]}, '

            elif param.type == "string":
                candidates = quoted_strings if quoted_strings else word_strings
                if candidates:
                    value = self.generate_string(crt_param_prompt, candidates)
                    if value in quoted_strings:
                        quoted_strings.remove(value)
                    elif value in word_strings:
                        word_strings.remove(value)
                else:
                    value = ""
                parameters[param_name] = value
                param_prompt += f'"{param_name}": "{value}", '

        return {
            "prompt": prompt,
            "name": generated_name,
            "parameters": parameters
        }

    def extract_numbers_from_prompt(self, prompt: str) -> list[str]:
        """Extract numeric candidates from the prompt."""
        return re.findall(r'-?\d+\.?\d*', prompt)

    def extract_quoted_strings(self, prompt: str) -> list[str]:
        """Extract strings explicitly wrapped in quotes from the prompt.

        Prioritizes double-quoted strings to avoid splitting on apostrophes
        (e.g. in "Hello I'm 233 years old"). Falls back to
        single-quoted strings.

        Args:
            prompt: The natural language prompt to extract from.

        Returns:
            List of quoted string contents, in order of appearance.
        """
        # Cas "Format template: <tout le reste>"
        template_match = re.findall(
            r'[Ff]ormat template[:\s]+(.+)$', prompt
        )
        if template_match:
            return [template_match[0].strip()]

        # Guillemets doubles en priorité
        quoted = re.findall(r'"([^"]+)"', prompt)
        if not quoted:
            quoted = re.findall(r"'([^']+)'", prompt)

        # Chemins Unix et Windows non quotés
        unix = re.findall(r'/[\w./\-]+', prompt)
        windows = re.findall(r'[A-Za-z]:\\[\w\\.\-]+', prompt)
        paths = unix + windows

        return quoted + [p for p in paths if p not in quoted]

    def extract_word_strings(self, prompt: str) -> list[str]:
        """Extract individual non-stopword words from the prompt.

        Used as fallback candidates for inferred parameters such as regex
        patterns or replacement values that are not quoted in the prompt.

        Args:
            prompt: The natural language prompt to extract from.

        Returns:
            List of words not present in the stopword list,
            preserving original case.
        """

        STOPWORDS = {
            "what", "is", "the", "a", "an", "of", "in",
            "and", "greet", "reverse", "string", "replace",
            "all", "with", "function", "call", "using",
            "substitute", "word", "format", "template",
            "read", "file", "at", "on", "run", "execute",
            "query", "calculate", "compound", "interest",
        }
        words = re.findall(r"[a-zA-Z]+", prompt)
        result = [w for w in words
                if len(w) > 1 and w.lower() not in STOPWORDS]

        # Encodings avec tiret : utf-8, latin-1, ascii, etc.
        encodings = re.findall(r'[a-zA-Z][\w]*-\d+', prompt)
        return result + [e for e in encodings if e not in result]

    def get_valid_tokens_for_nb(self,
                                generated_so_far: str,
                                candidates: list[str]) -> list[int]:
        """Return valid token IDs for the next number token.

        Args:
            generated_so_far: The number string generated so far.
            candidates: List of numeric string candidates
                        extracted from the prompt.

        Returns:
            List of token IDs that would continue
            at least one candidate number.
        """
        valid_ids = []

        for number in candidates:
            if not number.startswith(generated_so_far):
                continue

            number_token_ids = self.model.encode(number)[0].tolist()
            pos = len(self.model.encode(generated_so_far)[0].tolist()) \
                if generated_so_far else 0

            if pos < len(number_token_ids):
                valid_ids.append(number_token_ids[pos])

        return list(set(valid_ids))

    def get_valid_tokens_for_string(self,
                                    generated_so_far: str,
                                    candidates: list[str]) -> list[int]:
        """Return valid token IDs for the next string token.

        Args:
            generated_so_far: The string generated so far,
                              reconstructed via decode.
            candidates: List of string candidates extracted from the prompt.

        Returns:
            List of token IDs that would continue
            at least one candidate string.
        """
        valid_ids = []

        for number in candidates:
            if not number.startswith(generated_so_far):
                continue

            number_token_ids = self.model.encode(number)[0].tolist()
            pos = len(self.model.encode(generated_so_far)[0].tolist()) \
                if generated_so_far else 0

            if pos < len(number_token_ids):
                valid_ids.append(number_token_ids[pos])

        return list(set(valid_ids))

    def generate_number(self, full_prompt: str, candidates: list[str]) -> str:
        """Generate a number value by constrained decoding
        over numeric candidates.

        Iterates token by token, restricting choices to
        tokens that keep at least one candidate alive as a prefix match.
         Stops when a full candidate is matched.

        Args:
            full_prompt: The full context string passed to the LLM.
            candidates: List of numeric string candidates
                        to constrain decoding to.

        Returns:
            The matched numeric string, or
            an empty string if no candidate matched.
        """
        generated_ids: list[int] = []

        while True:
            generated = self.model.decode(generated_ids) if generated_ids else ""
            valid_ids = self.get_valid_tokens_for_nb(generated, candidates)

            if not valid_ids:
                break

            current_text = full_prompt + generated
            input_ids = self.model.encode(current_text)[0].tolist()
            logits = self.model.get_logits_from_input_ids(input_ids)

            masked = [-math.inf] * len(logits)
            for token_id in valid_ids:
                masked[token_id] = logits[token_id]

            next_token_id = masked.index(max(masked))
            generated_ids.append(next_token_id)
            generated = self.model.decode(generated_ids)

            if any(generated == n for n in candidates):
                break

        return self.model.decode(generated_ids) if generated_ids else ""
        

    def generate_string(self, full_prompt: str, candidates: list[str]) -> str:
        """Generate a string value by constrained decoding
           over string candidates.

        Uses token IDs internally and reconstructs
        the string via model.decode()
        at each step to avoid BPE artefacts (e.g. leading Ġ characters).
        Stops when a full candidate is matched.

        Args:
            full_prompt: The full context string passed to the LLM.
            candidates: List of string candidates to constrain decoding to.

        Returns:
            The matched string, or an empty string if no candidate matched.
        """
        generated_ids: list[int] = []

        while True:
            generated = self.model.decode(generated_ids) \
                if generated_ids else ""

            valid_ids = self.get_valid_tokens_for_string(generated, candidates)

            if not valid_ids:
                break

            current_text = full_prompt + generated
            input_ids = self.model.encode(current_text)[0].tolist()
            logits = self.model.get_logits_from_input_ids(input_ids)

            masked = [-math.inf] * len(logits)
            for token_id in valid_ids:
                masked[token_id] = logits[token_id]

            next_token_id = masked.index(max(masked))
            generated_ids.append(next_token_id)
            generated = self.model.decode(generated_ids)

            if any(generated == w for w in candidates):
                break

        return self.model.decode(generated_ids) if generated_ids else ""
