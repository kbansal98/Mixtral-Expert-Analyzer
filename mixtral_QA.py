from mixtral_analyzer import MixtralExpertTracker
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

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

    # Analyze the sample
    sample_usage, response = tracker.analyze_sample(text)
    if sample_usage is None or sample_usage.size == 0:
        print(f"Sample {i + 1}: No expert usage data.")
        continue

    # Append the sample usage to the list
    all_sample_usages.append(sample_usage)

    # Compute and print per-sample average expert utilization
    sample_mean = sample_usage.mean(axis=0)  # Average across layers for this sample
    print(f"Expert utilization for Sample {i + 1}:")
    for expert_idx, usage in enumerate(sample_mean):
        print(f"  Expert {expert_idx}: {usage * 100:.2f}%")
    print("\nGenerated Response:", response)
    print("-" * 50)

# Convert results to a NumPy array for further analysis
all_sample_usages = np.array(all_sample_usages)
print(f"\nAccumulated results shape: {all_sample_usages.shape}")

# Compute and print overall statistics across all samples
print("\nOverall Expert Utilization Statistics (averaged across all samples):")
mean_expert_usage = all_sample_usages.mean(axis=(0, 1))  # Average across samples and layers
for expert_idx, usage in enumerate(mean_expert_usage):
    print(f"  Expert {expert_idx}: {usage * 100:.2f}%")