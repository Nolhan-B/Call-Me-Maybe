def decode(prompt: str, model, vocab, functions) -> dict:
    
	ids = model.encode(prompt)
	generated = ""
	state = "start"
	return {}