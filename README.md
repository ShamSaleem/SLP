---
license: mit
datasets:
- Locutusque/InstructMix
language:
- en
metrics:
- bleu
- perplexity
pipeline_tag: text-generation
widget:
  - text: '<|USER|> Design a Neo4j database and Cypher function snippet to Display Extreme Dental hygiene: Using Mouthwash for Analysis for Beginners. Implement if/else or switch/case statements to handle different conditions related to the Consent. Provide detailed comments explaining your control flow and the reasoning behind each decision. <|ASSISTANT|> '
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->
This a fine-tuned version of gpt2 on Locutusque/InstructMix.

## Model Details
This model performs significantly better than Locutusque/gpt2-large-conversational. Here are the training results:


- BLEU - 30
- Perplexity - 5
### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** Locutusque
- **Shared by [optional]:** [More Information Needed]
- **Model type:** GPT-2
- **Language(s) (NLP):** English
- **License:** mit
- **Finetuned from model [optional]:** GPT-2

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
This model is designed to follow instructions, or partake in conversations.

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Instruction-following or conversational.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model struggles to write complex code, and I only recommend simple code from this model.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This model will most likely produce false information, especially about history. Make sure to confirm the responses this model makes.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

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

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

https://huggingface.co/datasets/Locutusque/InstructMix

This model has so far been trained on 600,000 examples of the linked data, with more training sessions to come.

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** fp16 non-mixed precision <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- BLEU = 30
- Perplexity = 5

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]