import sys
from typing import Dict


class Parser:
    """Utility class for parsing command-line arguments."""

    @staticmethod
    def get_dict_config_from_args() -> Dict[str, str]:
        """Parse command-line arguments into a configuration dictionary.

        Reads sys.argv and maps known flags to internal config keys.
        Arguments are expected in pairs: --flag value.

        Returns:
            A dict with any of the following keys: 'fct_file',
            'input_file', 'output_file', depending on which flags
            were provided.
        """
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
