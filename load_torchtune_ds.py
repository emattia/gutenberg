# load_torchtune_ds.py

import json
import os
import re
import tempfile
from typing import Any, Callable, Dict, Optional

from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchtune.datasets import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.dev.grpo.data import RLDataset, padded_collate_rl

# TODO: load these from download_src_data? common config?
VALID_ERAS = [
    "renaissance",
    "enlightenment",
    "romatnic",
    "victorian",
    "edwardian",
    "modern"
]

GUTENBERG_ERAS_PREAMBLE_PROMPT = (
    "A passage is fed to a language-analysis assistant. "
    "You, the assistant, first think about the nature of the text in the mind, then respond. "
    "The thinking should entail logic that supports the answers you respond with. "
    "Structure the contents of the thinking tag in clear statements and logical connectors between them. "
    "Responses should be EXACTLY in this XML format:\n"
    "<think>Your detailed reasoning process here...</think> "
    "<answer_date>YEAR</answer_date> "
    "<answer_era>ERA</answer_era>\n\n"
    f"ERA must be one of: {', '.join(VALID_ERAS)}. "
    "YEAR must be a number only. "
    "Do not include ANY text outside these three tags. "
    "\n\nExample of correct response:\n"
    "<think> This passage uses formal language and references to Victorian customs like afternoon tea. "
    "The characters discuss social obligations typical of 19th century England. "
    "Therefore, based on the literary style and social references, this text appears to be from the late Victorian period.</think> "
    "<answer_date>1880</answer_date> "
    "<answer_era>victorian</answer_era>"
    "\n\nIdentify the historical era and approximate date of the following text passage: {passage} "
    "Assistant: "
)

TRAINABLE_PROMPT = "<think>{cot}</think> <answer_date>{date}</answer_date> <answer_era>{era}</answer_era>"


def transform_gutenberg_instance(problem: dict[str, str]) -> dict[str, str]:
    """
    Parses an item from the historical context dataset into a ReasoningProblem
    by extracting the passage, reasoning, predicted date and predicted era.

    Args:
        problem: A dictionary containing passage data
    Returns:
        A dictionary with question, cot, answer_era, and answer_date
    """
    passage = problem["passage"]
    era = problem["era"]
    date = problem.get("date", "")  # Get date with fallback
    
    # If date is missing, estimate from era
    if not date:
        era_dates = {
            "renaissance": "1575",
            "enlightenment": "1725",
            "victorian": "1870",
            "edwardian": "1910",
            "modern": "1940"
        }
        date = era_dates.get(era.lower(), "1900")
    
    # Use the clues and rationale as the chain-of-thought reasoning
    # If they're empty, we'll create a placeholder
    clues = ", ".join(problem.get("clues", []))
    rationale = problem.get("rationale", "")
    
    if clues and rationale:
        cot = f"This passage contains several clues about its historical era. {clues}. {rationale}"
    else:
        cot = f"I need to analyze the language, references, and style of this passage to determine its era. " \
              f"The passage appears to be from the {era} era (around {date}) based on its style, vocabulary, and references."
    
    return {"question": passage, "cot": cot, "answer_era": era, "answer_date": date}


class GutenbergErasRLDataset(RLDataset):
    def __init__(self, dataset, problem_transform, tokenizer):
        self._problem_transform = problem_transform
        self._tokenizer = tokenizer
        self._data = dataset
        
    def _prepare_sample(self, sample: dict) -> dict:
        transformed_sample = self._problem_transform(sample)
        
        question = GUTENBERG_ERAS_PREAMBLE_PROMPT.format(passage=transformed_sample["question"])
        
        q_tokens = self._tokenizer.encode(question, add_eos=False)
        mask = [1 for _ in q_tokens]
        
        # Extract both era and date
        answer_era = transformed_sample["answer_era"]
        answer_date = transformed_sample["answer_date"]
        
        # Return both, but also include a combined "answer" for compatibility
        return {
            "tokens": q_tokens, 
            "mask": mask, 
            "answer": f"{answer_era} ({answer_date})",
            "answer_era": answer_era,
            "answer_date": answer_date
        }
    

def load_gutenberg_dataset(
    tokenizer: ModelTokenizer,
    *,
    data_path: str = "gutenberg_dataset",
    file_pattern: str = "*.json",
    filter_fn: Optional[Callable] = None,
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> RLDataset:
    """
    Historical Context Reasoning dataset prepared for RL-based training with verifiable rewards.
    """
    import glob
    
    # Create a temporary directory for the dataset
    temp_dir = tempfile.mkdtemp(prefix="historical_dataset_")
    
    # Find all JSON files in the directory
    if os.path.isdir(data_path):
        json_files = glob.glob(os.path.join(data_path, file_pattern), recursive=True)
    else:
        # Assume it's a single file
        json_files = [data_path]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {data_path} matching pattern {file_pattern}")
    
    # Load all passage data from all files
    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    print(f"Loaded {len(all_data)} passages from {len(json_files)} files")
    
    # Convert to Dataset
    hf_dataset = Dataset.from_list(all_data)
    
    # Save to disk in a format that load_from_disk can read
    dataset_path = os.path.join(temp_dir, "historical_dataset")
    hf_dataset.save_to_disk(dataset_path)
    
    # Load the dataset using load_from_disk
    hf_dataset = load_from_disk(dataset_path)
    
    def default_filter_fn(example: dict, idx: int):
        if partition is None:
            return True

        match = re.match(r"^(\d+)-(\d+)/(\d+)$", partition)
        if not match:
            raise ValueError(
                f"Invalid partition format: {partition}. Expected format: start-end/total"
            )

        start, end, total = map(int, match.groups())

        current = idx % total
        return start <= current <= end

    actual_filter_fn = filter_fn if filter_fn is not None else default_filter_fn
    
    # Apply filtering if needed
    if filter_fn is not None:
        hf_dataset = hf_dataset.filter(actual_filter_fn, with_indices=True)
            
    return GutenbergErasRLDataset(hf_dataset, transform_gutenberg_instance, tokenizer)