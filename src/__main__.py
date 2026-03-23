import json
from llm_sdk import Small_LLM_Model
from src.repositories import FunctionRepository, PromptRepository
from src.parsing.parser import Parser
from src.parsing.config import ConfigModel


def main() -> None:

    config = ConfigModel(**Parser.get_dict_config_from_args())
    print(config)
    functions = FunctionRepository(filepath=config.fct_file).get_all()
    prompts = PromptRepository(filepath=config.input_file).get_all()
    model = Small_LLM_Model()
    path_to_vocab = model.get_path_to_vocab_file()

    with open(path_to_vocab, "r") as file:
        vocab = json.load(file)

    id_to_token = {value: key for key, value in vocab.items()}

    ids = model.encode("fn_add_numbers")
    token = [id_to_token[i] for i in ids[0].tolist()]
    for tokens in token:
        print(tokens)

    # for func in functions:
    #     ids = model.encode(func.name)
    #     tokens = [id_to_token[i] for i in ids[0].tolist()]
    #     print(f"{func.name} -> {tokens}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Erreur :", e)
