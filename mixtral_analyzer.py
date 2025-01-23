from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import torch
import numpy as np

class MixtralExpertTracker:
    def __init__(self):
        self.model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Ensure pad_token_id is set
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        # Explicitly set `num_experts` and `top_k`
        self.num_experts = 8
        self.top_k = 2

    def compute_expert_routing(self, router_logits, attention_mask=None):
        """Compute routing probabilities and tokens per expert."""
        if router_logits is None:
            return None

        # Compute routing weights
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

        return tokens_per_expert.detach().cpu().numpy()

    def analyze_sample(self, text):
        """Analyze a single sample for expert utilization and model response."""
        # Prepare prompt
        prompt = f"[INST] {text} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(0)

        with torch.no_grad():
            # Run inference and enable router logits output
            outputs = self.model(
                **inputs,
                output_router_logits=True,
                return_dict=True
            )

        # Collect router logits and compute expert utilization
        layer_expert_usages = []
        if hasattr(outputs, "router_logits"):
            for layer_idx, layer_logits in enumerate(outputs.router_logits):
                tokens_per_expert = self.compute_expert_routing(
                    layer_logits,
                    attention_mask=inputs.get("attention_mask", None)
                )
                if tokens_per_expert is not None:
                    layer_expert_usages.append(tokens_per_expert)

        # Generate response
        output = self.model.generate(**inputs, max_new_tokens=50)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return np.array(layer_expert_usages), response