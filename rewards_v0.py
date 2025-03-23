
from xml.etree import ElementTree as ET
from typing import Tuple, List, Dict, Any, Optional
import torch
import re
from torchtune.modules.transforms.tokenizers import ModelTokenizer

# Define valid eras for validation
VALID_ERAS = [
    "renaissance",
    "enlightenment",
    "victorian",
    "edwardian",
    "modern"
]

def check_outside_text(text: str) -> tuple[bool, str]:
    """
    More robust function to detect text outside the required tags.

    Args:
        text: The text to analyze

    Returns:
        tuple: (has_outside_text, outside_text)
    """
    # First strip whitespace
    text = text.strip()

    # Use regex to extract all tag content
    pattern = r"<(think|answer_date|answer_era)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)

    # Create a cleaned version for comparison
    cleaned_text = text

    # Remove all valid tag content
    for tag, content in matches:
        cleaned_text = cleaned_text.replace(f"<{tag}>{content}</{tag}>", "", 1)

    # Strip whitespace again
    cleaned_text = cleaned_text.strip()

    return bool(cleaned_text), cleaned_text


def extract_tags(text: str) -> dict[str, list[str]]:
    """
    Parse XML-like tags from text, with improved handling for malformed XML.

    Args:
        text: Text potentially containing XML tags

    Returns:
        Dictionary with tag content keyed by tag name
    """
    results = {
        "think": [],
        "answer_era": [],
        "answer_date": []
    }

    # First, try regex method (more robust for malformed XML)
    for tag in ["think", "answer_era", "answer_date"]:
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Trim whitespace from each match
            results[tag] = [match.strip() for match in matches]

    # If regex found matches, return those
    if any(len(v) > 0 for v in results.values()):
        return results

    # Otherwise try the XML parser as fallback
    xml_string = f"<root>{text}</root>"
    try:
        root = ET.fromstring(xml_string)
        for tag in ["think", "answer_era", "answer_date"]:
            results[tag] = [
                (elem.text.strip() if elem.text else "") 
                for elem in root.findall(tag)
            ]
        return results
    except ET.ParseError:
        # Return the empty results if both methods fail
        return results


def shaped_correctness_reward(answer: str, completion: str) -> tuple[float, float, Dict[str, Any]]:
    """
    Enhanced reward function for historical era identification with detailed diagnostics.

    Args:
        answer: Ground-truth answer string in format "era (date)"
        completion: Model's completion string

    Returns:
        tuple: (reward_score, success_flag, details_dict)
    """
    reward = 0.0
    success = 0.0

    # Create details dictionary for diagnostics
    details = {
        "ground_truth": {
            "original": answer,
            "era": "",
            "date": ""
        },
        "completion": completion[:100] + ("..." if len(completion) > 100 else ""),  # Truncated for readability
        "extracted_tags": {},
        "format_analysis": {},
        "content_analysis": {},
        "reward_components": [],
        "total_reward": 0.0,
        "success": 0.0
    }

    # Parse the ground truth era and date
    gt_match = re.match(r'([a-z]+)\s*\((\d+)\)', answer.lower())
    if gt_match:
        gt_era = gt_match.group(1).strip()
        gt_date = gt_match.group(2).strip()
    else:
        # Fallback if parsing fails
        gt_era = answer.lower().strip()
        gt_date = ""

    # Store ground truth values in details
    details["ground_truth"]["era"] = gt_era
    details["ground_truth"]["date"] = gt_date

    # Extract tags from completion
    tags = extract_tags(completion)
    details["extracted_tags"] = {
        "think": tags["think"],
        "answer_era": tags["answer_era"],
        "answer_date": tags["answer_date"]
    }

    # Check for text outside tags - improved method
    has_outside_text, outside_text = check_outside_text(completion)
    details["format_analysis"]["has_outside_text"] = has_outside_text
    if has_outside_text:
        details["format_analysis"]["outside_text"] = outside_text[:100] + ("..." if len(outside_text) > 100 else "")

    # Apply significant penalties for text outside tags
    if has_outside_text:
        # More aggressive penalty for text outside tags
        penalty = min(30.0, len(outside_text) * 0.2)
        reward -= penalty
        details["reward_components"].append({
            "component": "outside_text_penalty",
            "value": -penalty,
            "reason": f"Text found outside required tags: '{outside_text[:30]}...' ({len(outside_text)} chars)"
        })

    # Format rewards - higher values for proper formatting
    if len(tags["think"]) == 1:
        reward += 10.0  # Good reward for having exactly one thinking section
        details["reward_components"].append({
            "component": "think_tag_format",
            "value": 10.0,
            "reason": "Correct: Exactly one <think> tag"
        })
    elif len(tags["think"]) > 1:
        reward += 2.5   # Smaller reward for having thinking, but too many sections
        details["reward_components"].append({
            "component": "think_tag_format",
            "value": 2.5,
            "reason": f"Partial: {len(tags['think'])} <think> tags found (expected 1)"
        })
    else:
        reward -= 5.0   # Penalty for missing thinking section
        details["reward_components"].append({
            "component": "think_tag_format",
            "value": -5.0,
            "reason": "Missing <think> tag"
        })

    if len(tags["answer_era"]) == 1:
        reward += 10.0  # Increased reward for having exactly one era tag
        details["reward_components"].append({
            "component": "era_tag_format",
            "value": 10.0,
            "reason": "Correct: Exactly one <answer_era> tag"
        })
    elif len(tags["answer_era"]) > 1:
        reward += 2.5   # Some reward for having era tags, but too many
        details["reward_components"].append({
            "component": "era_tag_format",
            "value": 2.5,
            "reason": f"Partial: {len(tags['answer_era'])} <answer_era> tags found (expected 1)"
        })
    else:
        reward -= 10.0  # Stronger penalty for missing era
        details["reward_components"].append({
            "component": "era_tag_format",
            "value": -10.0,
            "reason": "Missing <answer_era> tag"
        })

    if len(tags["answer_date"]) == 1:
        reward += 10.0  # Increased reward for having exactly one date tag
        details["reward_components"].append({
            "component": "date_tag_format",
            "value": 10.0,
            "reason": "Correct: Exactly one <answer_date> tag"
        })
    elif len(tags["answer_date"]) > 1:
        reward += 2.5   # Some reward for having date tags, but too many
        details["reward_components"].append({
            "component": "date_tag_format",
            "value": 2.5,
            "reason": f"Partial: {len(tags['answer_date'])} <answer_date> tags found (expected 1)"
        })
    else:
        reward -= 10.0  # Stronger penalty for missing date
        details["reward_components"].append({
            "component": "date_tag_format",
            "value": -10.0,
            "reason": "Missing <answer_date> tag"
        })

    # Era validation rewards
    if tags["answer_era"]:
        # Store era analysis
        details["content_analysis"]["era"] = {
            "provided": [era.lower() for era in tags["answer_era"]],
            "valid_eras": VALID_ERAS,
            "ground_truth": gt_era
        }

        # Check if any provided era is in the valid list
        valid_provided = [era.lower() for era in tags["answer_era"] if era.lower() in VALID_ERAS]
        if valid_provided:
            reward += 5.0  # Bonus for using a valid era from the list
            details["reward_components"].append({
                "component": "era_validation",
                "value": 5.0,
                "reason": f"Used valid era(s): {', '.join(valid_provided)}"
            })
        else:
            reward -= 5.0  # Penalty for using invalid era
            details["reward_components"].append({
                "component": "era_validation",
                "value": -5.0,
                "reason": f"Invalid era(s): {', '.join([era.lower() for era in tags['answer_era']])}"
            })

    # Correctness rewards for era
    if tags["answer_era"]:
        exact_match = any(gt_era == attempt.lower().strip() for attempt in tags["answer_era"])
        partial_match = any(gt_era in attempt.lower().strip() for attempt in tags["answer_era"])

        details["content_analysis"]["era_match"] = {
            "exact_match": exact_match,
            "partial_match": partial_match
        }

        if exact_match:
            # One of the answer_era tags has the exact right era
            reward += 30.0  # Increased reward for correct era
            details["reward_components"].append({
                "component": "era_correctness",
                "value": 30.0,
                "reason": f"Correct era: {gt_era}"
            })
        elif partial_match:
            # One of the answer_era tags contains the right era as a substring
            reward += 10.0  # Partial reward
            details["reward_components"].append({
                "component": "era_correctness",
                "value": 10.0,
                "reason": f"Partial era match: Contains '{gt_era}'"
            })
        else:
            details["reward_components"].append({
                "component": "era_correctness",
                "value": 0.0,
                "reason": f"Incorrect era: Expected '{gt_era}'"
            })

    # Correctness rewards for date
    if gt_date and tags["answer_date"]:
        try:
            gt_year = int(gt_date)

            # Parse all dates and track errors
            date_attempts = []
            valid_dates = []

            for attempt in tags["answer_date"]:
                attempt = attempt.strip()
                date_attempts.append(attempt)
                if attempt.isdigit():
                    valid_dates.append(int(attempt))

            details["content_analysis"]["date"] = {
                "provided": date_attempts,
                "valid_dates": valid_dates,
                "ground_truth": gt_year
            }

            if valid_dates:
                # Find best date attempt (closest to ground truth)
                best_diff = min(abs(date - gt_year) for date in valid_dates)
                best_date = next(date for date in valid_dates if abs(date - gt_year) == best_diff)

                details["content_analysis"]["date"]["best_match"] = {
                    "value": best_date,
                    "difference": best_diff
                }

                # Award based on closest date
                if best_diff == 0:
                    # Exact date match
                    reward += 30.0  # Increased reward
                    details["reward_components"].append({
                        "component": "date_correctness",
                        "value": 30.0,
                        "reason": f"Exact date match: {best_date}"
                    })
                elif best_diff <= 20:
                    # Within 20 years
                    reward += 20.0  # Increased reward
                    details["reward_components"].append({
                        "component": "date_correctness",
                        "value": 20.0,
                        "reason": f"Close date match: {best_date} (within 20 years of {gt_year})"
                    })
                elif best_diff <= 50:
                    # Within 50 years
                    reward += 10.0
                    details["reward_components"].append({
                        "component": "date_correctness",
                        "value": 10.0,
                        "reason": f"Approximate date: {best_date} (within 50 years of {gt_year})"
                    })
                elif best_diff <= 100:
                    # Within 100 years
                    reward += 5.0
                    details["reward_components"].append({
                        "component": "date_correctness",
                        "value": 5.0,
                        "reason": f"Distant date: {best_date} (within 100 years of {gt_year})"
                    })
                else:
                    # More than 100 years off
                    reward -= 5.0  # Small penalty for very wrong date
                    details["reward_components"].append({
                        "component": "date_correctness",
                        "value": -5.0,
                        "reason": f"Incorrect date: {best_date} (more than 100 years from {gt_year})"
                    })
            else:
                # No valid numeric dates found
                reward -= 5.0  # Penalty for non-numeric date
                details["reward_components"].append({
                    "component": "date_correctness",
                    "value": -5.0,
                    "reason": f"Non-numeric date(s): {', '.join(date_attempts)}"
                })

        except ValueError as e:
            # Penalty for non-numeric date
            reward -= 5.0
            details["reward_components"].append({
                "component": "date_correctness",
                "value": -5.0,
                "reason": f"Date parsing error: {str(e)}"
            })

    # Full success criteria - more strict
    # Both era and date must be correct AND format must be perfect
    perfect_format = (
        len(tags["think"]) == 1 and 
        len(tags["answer_era"]) == 1 and 
        len(tags["answer_date"]) == 1 and
        not has_outside_text
    )

    correct_era = (
        tags["answer_era"] and 
        tags["answer_era"][0].lower().strip() == gt_era
    )

    correct_date = (
        gt_date and
        tags["answer_date"] and
        tags["answer_date"][0].isdigit() and 
        abs(int(tags["answer_date"][0]) - int(gt_date)) <= 20
    )

    details["success_criteria"] = {
        "perfect_format": perfect_format,
        "correct_era": correct_era,
        "correct_date": correct_date
    }

    if perfect_format and correct_era and correct_date:
        reward = 100.0
        success = 1.0
        details["reward_components"].append({
            "component": "perfect_answer",
            "value": "100.0 (overwrites previous)",
            "reason": "Perfect format and correct answers"
        })

    # Store final reward and success in details
    details["total_reward"] = reward
    details["success"] = success

    return reward, success, details


def batch_shaped_correctness_reward(
    tokenizer: ModelTokenizer, 
    completions: torch.Tensor, 
    answers: List[str],
    details_report: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]:
    """
    Apply the shaped reward function to a batch of completions.

    Args:
        tokenizer: Tokenizer for decoding completions
        completions: Tensor of token IDs
        answers: List of ground truth answers
        details_report: Whether to generate detailed diagnostic reports

    Returns:
        Tuple of (rewards, successes, optional details list)
    """
    batch_size, grpo_size, *_ = completions.shape
    rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)

    # Create container for details if requested
    details_list = [] if details_report else None

    # Process each completion in the batch
    for b in range(batch_size):
        batch_details = [] if details_report else None

        for g in range(grpo_size):
            # Decode the completion
            text_completion = tokenizer.decode(
                completions[b, g].tolist()
            )

            # Calculate reward, success, and details
            reward, success, details = shaped_correctness_reward(
                answer=answers[b], 
                completion=text_completion
            )

            # Store results
            rewards[b, g] = reward
            successes[b, g] = success

            # Store details if requested
            if details_report:
                # Add batch and group indices
                details["batch_idx"] = b
                details["group_idx"] = g
                batch_details.append(details)

        # Add batch details to the main list
        if details_report:
            details_list.append(batch_details)

    return rewards, successes, details_list


# Helper function to print a readable summary of the details
def print_reward_details_summary(details: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the reward details.

    Args:
        details: The details dictionary from shaped_correctness_reward
    """
    print(f"=== Reward Calculation Summary ===")
    print(f"Ground Truth: Era='{details['ground_truth']['era']}', Date='{details['ground_truth']['date']}'")
    print(f"Completion: {details['completion']}")
    print("\nExtracted Tags:")
    print(f"  Think: {len(details['extracted_tags']['think'])} tag(s)")
    print(f"  Era: {len(details['extracted_tags']['answer_era'])} tag(s)")
    print(f"  Date: {len(details['extracted_tags']['answer_date'])} tag(s)")

    print("\nFormat Analysis:")
    if details['format_analysis'].get('has_outside_text', False):
        print(f"  ❌ Text outside tags: {details['format_analysis']['outside_text']}")
    else:
        print(f"  ✓ No text outside tags")

    print("\nContent Analysis:")
    if 'era' in details['content_analysis']:
        print(f"  Era provided: {details['content_analysis']['era']['provided']}")
        match_status = "❌ No match"
        if details['content_analysis'].get('era_match', {}).get('exact_match', False):
            match_status = "✓ Exact match"
        elif details['content_analysis'].get('era_match', {}).get('partial_match', False):
            match_status = "~ Partial match"
        print(f"  Era match: {match_status}")

    if 'date' in details['content_analysis']:
        print(f"  Date provided: {details['content_analysis']['date']['provided']}")
        if 'best_match' in details['content_analysis']['date']:
            best = details['content_analysis']['date']['best_match']
            print(f"  Best date: {best['value']} (diff: {best['difference']} years)")

    print("\nReward Components:")
    for component in details['reward_components']:
        print(f"  {component['component']}: {component['value']} - {component['reason']}")

    print(f"\nTotal Reward: {details['total_reward']}")
    print(f"Success: {details['success']}")

    if 'success_criteria' in details:
        criteria = details['success_criteria']
        print("\nSuccess Criteria:")
        print(f"  Format perfect: {'✓' if criteria['perfect_format'] else '❌'}")
        print(f"  Era correct: {'✓' if criteria['correct_era'] else '❌'}")
        print(f"  Date correct: {'✓' if criteria['correct_date'] else '❌'}")

    print("================================")
