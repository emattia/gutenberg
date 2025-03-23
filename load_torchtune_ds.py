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

# Configurable prompt templates
VALID_ERAS = [
    "renaissance",
    "enlightenment",
    "victorian",
    "edwardian",
    "modern"
]

GUTENBERG_ERAS_PREAMBLE_PROMPT = (
    "A passage is fed to a language-analysis assistant. "
    "You, the assistant, first thinks about the nature of the text in the mind, then responds. "
    "You, the assistant, must respond in EXACTLY this XML format:\n"
    "<think>Your detailed reasoning process here...</think> "
    "<answer_date>YEAR</answer_date> "
    "<answer_era>ERA</answer_era>\n\n"
    f"ERA must be one of: {', '.join(VALID_ERAS)}. "
    "YEAR must be a number only. "
    "Do not include ANY text outside these three tags. "
    "\n\nExample of correct response:\n"
    "<think>This passage uses formal language and references to Victorian customs like afternoon tea. "
    "The characters discuss social obligations typical of 19th century England. "
    "Based on the literary style and social references, this appears to be from the late Victorian period.</think> "
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


def sft_historical_transform(problem: dict[str, str]) -> dict[str, str]:
    """
    Prepares an item from the historical dataset into a format for SFT training.
    """
    passage = problem["passage"]
    era = problem["era"]
    date = problem.get("date", "")
    
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
    clues = ", ".join(problem.get("clues", []))
    rationale = problem.get("rationale", "")
    
    if clues and rationale:
        cot = f"This passage contains several clues about its historical era. {clues}. {rationale}"
    else:
        cot = f"I need to analyze the language, references, and style of this passage to determine its era. " \
              f"The passage appears to be from the {era} era (around {date}) based on its characteristics."
    
    preamble = GUTENBERG_ERAS_PREAMBLE_PROMPT.format(passage=passage)
    trainable = TRAINABLE_PROMPT.format(cot=cot, era=era, date=date)

    return {"preamble": preamble, "trainable": trainable}


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
        json_files = glob.glob(os.path.join(data_path, "**", file_pattern), recursive=True)
        # Filter out any JSON files in era subdirectories that aren't passage files
        json_files = [f for f in json_files if "_passage_" in os.path.basename(f)]
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
    
    # class GutenbergErasRLDataset(RLDataset):
    #     def __init__(self, dataset, problem_transform, tokenizer):
    #         self._problem_transform = problem_transform
    #         self._tokenizer = tokenizer
    #         self._data = dataset
            
    return GutenbergErasRLDataset(hf_dataset, transform_gutenberg_instance, tokenizer)


def historical_context_sft(
    tokenizer: ModelTokenizer,
    *,
    data_path: str = "gutenberg_dataset/passages_for_annotation_chunk_1.json",
    filter_fn: Optional[Callable] = None,
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    Historical Context Reasoning dataset prepared for SFT-based training.
    
    Args:
        tokenizer: The tokenizer to use for encoding text
        data_path: Path to the JSON file containing passage data
        filter_fn: Optional function to filter examples
        partition: Optional string in format "start-end/total" to use only a subset of data
        **load_dataset_kwargs: Additional arguments for dataset loading
        
    Returns:
        An SFTDataset for historical context reasoning
    """
    # Create a temporary directory for the dataset
    temp_dir = tempfile.mkdtemp(prefix="historical_sft_dataset_")
    
    # Load and save the dataset in Arrow format that HuggingFace can read
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to Dataset
    hf_dataset = Dataset.from_list(data)
    
    # Save to disk in a format that load_dataset can read
    dataset_path = os.path.join(temp_dir, "historical_sft_dataset")
    hf_dataset.save_to_disk(dataset_path)

    def model_transform(problem: dict[str, str]) -> dict[str, list[int]]:
        pre_tokens = tokenizer.encode(problem["preamble"], add_eos=False)
        trainable_tokens = tokenizer.encode(problem["trainable"], add_bos=False)

        # 1 == discard the token, 0 == include the token in training
        mask = [1 for t in pre_tokens] + [0 for t in trainable_tokens]

        return {"tokens": pre_tokens + trainable_tokens, "mask": mask}

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
    
    ds = SFTDataset(
        source=dataset_path,
        message_transform=sft_historical_transform,
        model_transform=model_transform,
        filter_fn=actual_filter_fn,
        filter_kwargs=dict(with_indices=True),
        **load_dataset_kwargs,
    )
    
    return ds


def create_historical_dataloader(
    tokenizer: ModelTokenizer,
    data_path: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    partition: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader for the historical context dataset.
    
    Args:
        tokenizer: The tokenizer to use
        data_path: Path to the JSON file containing passages
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        partition: Optional string in format "start-end/total" to use only a subset of data
        
    Returns:
        A DataLoader for the dataset
    """
    dataset = load_gutenberg_dataset(
        tokenizer,
        data_path=data_path,
        partition=partition,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: padded_collate_rl(
            batch,
            padding_idx=tokenizer.pad_id,
            ignore_idx=-100,  # CROSS_ENTROPY_IGNORE_IDX
        ),
    )
    
    return dataloader

    


def create_historical_sft_dataloader(
    tokenizer: ModelTokenizer,
    data_path: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    partition: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader for the historical context dataset for SFT training.
    
    Args:
        tokenizer: The tokenizer to use
        data_path: Path to the JSON file containing passages
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        partition: Optional string in format "start-end/total" to use only a subset of data
        
    Returns:
        A DataLoader for the dataset
    """
    dataset = historical_context_sft(
        tokenizer,
        data_path=data_path,
        partition=partition,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    
    return dataloader

