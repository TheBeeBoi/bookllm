from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path):
    # load pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    # generate text based on the given prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # specify the path to the pre-trained model
    model_path = './arxiv_papers/trained_model'

    # load pre-trained model and tokenizer
    model, tokenizer = load_model(model_path)

    # example prompt for text generation
    prompt = "Trains are"

    # generate text based on the prompt
    generated_text = generate_text(model, tokenizer, prompt, max_length=100)

    # print generated text
    print(generated_text)
