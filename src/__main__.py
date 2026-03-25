import json
from llm_sdk import Small_LLM_Model
from src.repositories import FunctionRepository, PromptRepository
from src.parsing.parser import Parser
from src.parsing.config import ConfigModel
from src.decoder import ConstrainedDecoder


def main() -> None:

    config = ConfigModel(**Parser.get_dict_config_from_args())
    print(config)
    functions = [
        f for f in FunctionRepository(filepath=config.fct_file).get_all()
        if f is not None
    ]

    prompts = [
        p for p in PromptRepository(filepath=config.input_file).get_all()
        if p is not None
    ]
    model = Small_LLM_Model()
    path_to_vocab = model.get_path_to_vocab_file()

    with open(path_to_vocab, "r") as file:
        vocab = json.load(file)

    id_to_token = {value: key for key, value in vocab.items()}

    decoder = ConstrainedDecoder(
        model=model,
        vocab=vocab,
        id_to_token=id_to_token,
        functions=functions
    )

    res = []

    for prompt in prompts:
        current = decoder.decode(prompt.prompt)
        print(current)
        res.append(current)

    with open(config.output_file, "w") as file:
        json.dump(res, file, indent=4)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Erreur :", e)
