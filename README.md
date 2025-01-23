# Mixtral Expert Analyzer
<img src="https://github.com/kbansal98/Mixtral-Expert-Analyzer/blob/main/Transformer-Encoder-with-MoE-Layers.jpg"/>
A program which tracks the token distribution across experts in the Mixtral 8x7B model. Can be used to track model efficiency for different types of inputs. More functionality such as energy usage will be added in the future.

The requirements for running the code depend on what you already have installed and what data you want to use, but a general list is shown below.

```python
pip install transformers
pip install tqdm
pip install datasets
pip install bitsandbytes
```
## Analyzer

The core of the program is the analyzer file, this is what handles the tracking of the token distribution in the Mixtral model. If one wants to add further metrics from the model to be analyzed this is where it should be done. 
Otherwise simply run the QA file and it will import the analyzer to be used on a dataset.

The tokens per expert are acquired from the logits of the model in the following way
```python
        routing_weights = torch.softmax(router_logits.to(torch.float16), dim=-1)

        # Get top-k experts per token
        top_k_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # Create one-hot expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, self.num_experts)

        if attention_mask is not None:
            # Reshape attention mask to match expert mask dimensions
            reshaped_mask = attention_mask.view(-1, 1, 1)
            expert_mask = expert_mask * reshaped_mask
            routing_weights = routing_weights * attention_mask.view(-1, 1)

        # Compute percentage of tokens routed to each expert
        tokens_per_expert = torch.mean(expert_mask.float(), dim=[0, 1])
```

In order to reduce computational load the code implements bitsandbytes 4 bit quantization by default, but this can be disabled here
```python
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto")
```

One can also change the maximum token generation from the model in the analyzer file
```python
output = self.model.generate(**inputs, max_new_tokens=50)
response = self.tokenizer.decode(output[0], skip_special_tokens=True)
```
## Dataset Usage

Any text based dataset can be used with this analyzer with minor modifications to the following code in the QA file, as an example the QA file utilizes the basicv8vc "SimpleQA" question answering dataset.

```python
dataset = load_dataset("basicv8vc/SimpleQA")

# Initialize tracker
tracker = MixtralExpertTracker()

# Initialize a list to store expert utilization for all samples
all_sample_usages = []

# Process samples
for i, sample in enumerate(tqdm(dataset["test"])):  # No split needed
    if i >= 10:  # Only process the first 10 samples
        break

    # Extract text from the dataset sample
    if "problem" in sample:  # Ensure the field is correct
        text = sample["problem"]
    else:
        print(f"Unexpected sample format at index {i}: {sample}")
        continue

    print(f"\nProcessing Sample {i + 1}: {text}")
```
