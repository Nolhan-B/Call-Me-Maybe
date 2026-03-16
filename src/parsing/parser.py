import sys
from typing import Dict
from pydantic import ValidationError
from src.parsing.config import ConfigModel


class Parser:

    @staticmethod
    def can_parse() -> bool:
        args = sys.argv[1:]

        if len(args) == 0:
            return False

        if len(args) % 2 != 0:
            return False

        valid_flags = {
            "--functions_definition",
            "--input",
            "--output"
        }

        for i in range(0, len(args), 2):
            if args[i] not in valid_flags:
                return False

        return True


    @staticmethod
    def get_dict_config_from_args() -> Dict[str, str]:
        args = sys.argv[1:]
        data = {}

        for i in range(0, len(args), 2):
            key = args[i]
            value = args[i + 1]

            match key:
                case "--functions_definition":
                    data["fct_file"] = value
                case "--input":
                    data["input_file"] = value
                case "--output":
                    data["output_file"] = value

        return data