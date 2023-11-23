Use the code below to get started with the model.

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large-conversational-retrain')
model = GPT2LMHeadModel.from_pretrained('gpt2-large-conversational-retrain')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def generate_text(model, tokenizer, prompt, max_length=1024):
    prompt = f'<|USER|> {prompt} <|ASSISTANT|> '
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    output = model.generate(input_ids, 
                            max_length=max_length, 
                            do_sample=True,
                            temperature=0.3, 
                            top_k=23, 
                            top_p=0.7,
                            repetition_penalty=1.176,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            attention_mask=attention_mask)
    output_ids = tokenizer.decode(output[0], skip_special_tokens=False)
    return output_ids
# Loop to interact with the model
while True:
    prompt = input("Enter a prompt (or 'q' to quit): ")
    if prompt == "q":
        break
    output_text = generate_text(model, tokenizer, prompt)
    print(output_text)
```

