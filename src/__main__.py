from src.parsing.parser import Parser
from src.parsing.config import ConfigModel

def main() -> None:
    if not Parser.can_parse():
        print("Invalid arguments")
        return

    pre_config = Parser.get_dict_config_from_args()
    configModel = ConfigModel(**pre_config)

    print(configModel)


if __name__ == "__main__":
    try:
        print()
        print()
        print()
        print()
        main()
        print()
        print()
        print()
    except Exception as e:
        print("unhandeled error caught :\n", e)