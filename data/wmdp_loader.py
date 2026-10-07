import os
import json
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict

DEFAULT_LOCAL_WMDP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wmdp")

class WMDPLoader:
    """
    DataLoader utility for the WMDP (Weapons of Mass Destruction Proxy) unlearning benchmark.
    Covers domain-specific safety & hazardous unlearning evaluation across:
    - wmdp-bio   (Biosecurity)
    - wmdp-cyber (Cybersecurity)
    - wmdp-chem  (Chemical security)
    """
    
    def __init__(self, local_dir: str = DEFAULT_LOCAL_WMDP):
        self.local_dir = local_dir

    def load_domain(self, domain: str = "wmdp-bio", use_local: bool = True) -> DatasetDict:
        """
        Loads a specific WMDP security domain dataset.
        
        Args:
            domain: 'wmdp-bio', 'wmdp-cyber', or 'wmdp-chem'
            use_local: If True, attempts to load from saved disk format first.
        """
        valid_domains = ["wmdp-bio", "wmdp-cyber", "wmdp-chem"]
        if domain not in valid_domains:
            raise ValueError(f"Domain '{domain}' invalid. Must be one of {valid_domains}")
            
        local_path = os.path.join(self.local_dir, domain)
        if use_local and os.path.exists(local_path):
            print(f"Loading WMDP '{domain}' from local disk: {local_path}")
            return DatasetDict.load_from_disk(local_path)
        else:
            print(f"Fetching WMDP '{domain}' from Hugging Face Hub (cais/wmdp)...")
            return load_dataset("cais/wmdp", domain)

    @staticmethod
    def format_multiple_choice(example: Dict) -> Tuple[str, int]:
        """
        Formats a WMDP multiple-choice question into a standardized prompt string and expected option index.
        """
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]
        
        letters = ["A", "B", "C", "D"]
        formatted_choices = "\n".join([f"({letters[i]}) {choices[i]}" for i in range(len(choices))])
        
        prompt = f"Question: {question}\nOptions:\n{formatted_choices}\nAnswer:"
        return prompt, answer_idx

if __name__ == "__main__":
    loader = WMDPLoader()
    print("Testing WMDP Loader...")
    try:
        ds_bio = loader.load_domain("wmdp-bio")
        test_split = ds_bio["test"]
        print(f"\nLoaded WMDP Bio test set: {len(test_split)} questions")
        sample_prompt, correct_idx = loader.format_multiple_choice(test_split[0])
        print("\nSample WMDP Multiple Choice Item:")
        print(sample_prompt)
        print(f"Correct Answer Index: {correct_idx}")
    except Exception as e:
        print(f"WMDP loader test failed (dataset might still be downloading): {e}")
