def inference(self, question: str, context: str = None, max_new_tokens: int = 2048):
    """
    Generate a response to the given question, using the provided context if available.

    Args:
        question (str): The user's question or query.
        context (str, optional): The relevant context information to be used for generating the response.
        max_new_tokens (int, optional): The maximum number of tokens to generate for the response.

    Returns:
        str: The generated response from the model.
    """
    if context:
        prompt = f"Using the information contained in the context, give a detailed answer to the question. Do not add any extra information.\n\nContext: {context}\n\nQuestion: {question}"
    else:
        prompt = f"Give a detailed answer to the question.\n\nQuestion: {question}"

    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

    with torch.no_grad():
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
