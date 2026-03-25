import sys
from typing import Dict


class Parser:

    @staticmethod
    def get_dict_config_from_args() -> Dict[str, str]:
        args = sys.argv[1:]
        data = {}

        for i in range(0, len(args) - 1, 2):
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
