import os
import json
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict

DEFAULT_LOCAL_MUSE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "muse")

class MUSELoader:
    """
    DataLoader utility for the MUSE (Machine Unlearning Six-Way Evaluation) benchmark.
    Tests unlearning, privacy leakage, scalability, and sequential unlearning across:
    - muse-news  (News articles domain)
    - muse-books (Literary books domain)
    """
    
    def __init__(self, local_dir: str = DEFAULT_LOCAL_MUSE):
        self.local_dir = local_dir

    def load_benchmark(self, domain: str = "muse-news", use_local: bool = True) -> DatasetDict:
        """
        Loads a specific MUSE benchmark dataset domain.
        
        Args:
            domain: 'muse-news' or 'muse-books'
            use_local: If True, attempts to load from saved disk format first.
        """
        local_path = os.path.join(self.local_dir, domain)
        if use_local and os.path.exists(local_path):
            print(f"Loading MUSE '{domain}' from local disk: {local_path}")
            return DatasetDict.load_from_disk(local_path)
        else:
            hf_repo = f"MUSE-Bench/MUSE-{domain.split('-')[-1].capitalize()}"
            print(f"Fetching MUSE dataset '{domain}' from Hugging Face Hub ({hf_repo})...")
            return load_dataset(hf_repo)

if __name__ == "__main__":
    loader = MUSELoader()
    print("Testing MUSE Loader...")
    try:
        ds_news = loader.load_benchmark("muse-news")
        print(f"\nLoaded MUSE News splits: {list(ds_news.keys())}")
    except Exception as e:
        print(f"MUSE loader test failed (dataset might still be downloading): {e}")
