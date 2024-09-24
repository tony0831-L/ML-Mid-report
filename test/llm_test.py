from transformers import AutoTokenizer, AutoModelForCausalLM


# fetch pretrain model from hugginface , use to generate response
# @model :hugginface model
# @max_length: respose length

def llm_test(model: str, max_length: int, prompt: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    response_hf_encoded = model.generate(
        user_input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )

    response_hf: str = tokenizer.decode(response_hf_encoded[:, user_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response_hf
