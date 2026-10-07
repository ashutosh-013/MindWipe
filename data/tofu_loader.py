import os
import json
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict

TOFU_DATASET_NAME = "locuslab/TOFU"
DEFAULT_LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tofu")

class TOFULoader:
    """
    DataLoader utility for TOFU (Task of Fictitious Unlearning) dataset.
    Supports loading from local disk storage or fetching directly from Hugging Face Hub.
    """
    
    def __init__(self, local_dir: str = DEFAULT_LOCAL_DIR):
        self.local_dir = local_dir

    def load_config(self, config_name: str = "forget10", use_local: bool = True) -> DatasetDict:
        """
        Loads a specific TOFU configuration (e.g. 'forget10', 'retain90', 'full').
        
        Args:
            config_name: Name of TOFU config ('forget10', 'retain90', 'full', 'forget01', etc.)
            use_local: If True, attempts to load from saved disk format first.
        
        Returns:
            DatasetDict containing splits for the requested configuration.
        """
        local_config_path = os.path.join(self.local_dir, config_name)
        
        if use_local and os.path.exists(local_config_path):
            print(f"Loading '{config_name}' from local disk: {local_config_path}")
            return DatasetDict.load_from_disk(local_config_path)
        else:
            print(f"Fetching '{config_name}' configuration from Hugging Face Hub ({TOFU_DATASET_NAME})...")
            return load_dataset(TOFU_DATASET_NAME, config_name)

    def load_experiment_pair(self, forget_config: str = "forget10", retain_config: str = "retain90") -> Tuple[Dataset, Dataset]:
        """
        Loads paired Forget and Retain datasets for CASU unlearning experiments.
        Default pair: 'forget10' (10% target to unlearn) and 'retain90' (90% target to preserve).
        
        Returns:
            Tuple of (forget_dataset, retain_dataset)
        """
        forget_ds_dict = self.load_config(forget_config)
        retain_ds_dict = self.load_config(retain_config)
        
        forget_ds = forget_ds_dict.get("train", forget_ds_dict[list(forget_ds_dict.keys())[0]])
        retain_ds = retain_ds_dict.get("train", retain_ds_dict[list(retain_ds_dict.keys())[0]])
        
        print(f"\n---> Dataset Pair Loaded:")
        print(f"   Forget Set ('{forget_config}'): {len(forget_ds)} examples")
        print(f"   Retain Set ('{retain_config}'): {len(retain_ds)} examples")
        
        return forget_ds, retain_ds

    @staticmethod
    def format_qa_prompt(question: str, answer: Optional[str] = None) -> str:
        """
        Formats question-answer pair into standard instruction prompt.
        """
        prompt = f"Question: {question}\nAnswer:"
        if answer:
            prompt += f" {answer}"
        return prompt

    def get_summary_statistics(self, config_name: str = "forget10") -> Dict:
        """
        Returns basic statistics for a given TOFU configuration.
        """
        ds_dict = self.load_config(config_name)
        stats = {}
        for split, ds in ds_dict.items():
            questions = [row.get("question", "") for row in ds]
            answers = [row.get("answer", "") for row in ds]
            avg_q_len = sum(len(q.split()) for q in questions) / max(len(questions), 1)
            avg_a_len = sum(len(a.split()) for a in answers) / max(len(answers), 1)
            
            stats[split] = {
                "num_samples": len(ds),
                "avg_question_word_count": round(avg_q_len, 2),
                "avg_answer_word_count": round(avg_a_len, 2),
                "sample_question": questions[0] if questions else "",
                "sample_answer": answers[0] if answers else ""
            }
        return stats

if __name__ == "__main__":
    loader = TOFULoader()
    print("Testing TOFU Loader for configuration 'forget10'...")
    try:
        forget_ds, retain_ds = loader.load_experiment_pair("forget10", "retain90")
        print("\nSample Forget QA Pair:")
        print(loader.format_qa_prompt(forget_ds[0]["question"], forget_ds[0]["answer"]))
        print("\nSample Retain QA Pair:")
        print(loader.format_qa_prompt(retain_ds[0]["question"], retain_ds[0]["answer"]))
    except Exception as e:
        print(f"Loader test failed: {e}")
