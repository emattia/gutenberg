import requests
import json
import os
import random
import time
import re
from tqdm import tqdm

# NOTE: Deliberately contentious choice.
# The idea is to demonstrate how to get an LLM to learn in a way that is culturally relevant to the use case.
# There are overlapping dates, names of eras, and people in other countries often wouldn't use these Euro-centric labels. 
# The idea is to show how GRPO-like methods are useful in moving an LLM towards a "sovereign AI" approach.
ERAS = {
    "renaissance": (1500, 1650),
    "enlightenment": (1650, 1800),
    "victorian": (1837, 1901),
    "edwardian": (1901, 1920),
    "modern": (1920, 1960)
}

BASE_DIR = "gutenberg_dataset"

def create_directories():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}/full", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/train", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/validation", exist_ok=True)
    for era in ERAS:
        os.makedirs(f"{BASE_DIR}/{era}", exist_ok=True)

def extract_date_hints(text, era_range):
    start_year, end_year = era_range
    year_pattern = r'\\b(1[5-9]\\d{2}|20[0-5]\\d)\\b'
    years = [int(y) for y in re.findall(year_pattern, text) if start_year <= int(y) <= end_year]
    return int(sorted(years)[len(years)//2]) if years else (start_year + end_year) // 2

def download_book_content(formats):
    for fmt in ["text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain", "text/html"]:
        if fmt in formats:
            try:
                response = requests.get(formats[fmt])
                if response.status_code == 200:
                    return response.text
            except:
                continue
    return None

def get_books_by_era(start_year, end_year, max_books):
    books, page = [], 1
    with tqdm(total=max_books, desc=f"Gathering books ({start_year}-{end_year})") as pbar:
        while len(books) < max_books:
            res = requests.get(f"https://gutendex.com/books/?page={page}").json()
            if not res.get("results"):
                break
            for book in res["results"]:
                if "en" not in book["languages"] or not book["authors"]:
                    continue
                author = book["authors"][0]
                if not author["birth_year"] or not author["death_year"]:
                    continue
                est_year = min(author["birth_year"] + 40, author["death_year"])
                if start_year <= est_year <= end_year:
                    book["estimated_date"] = est_year
                    books.append(book)
                    pbar.update(1)
                    if len(books) >= max_books:
                        break
            page += 1
            time.sleep(0.5)
    return books

def extract_passages(text, min_length=200, max_length=400, num_passages=10):
    """Extract several coherent passages from Project Gutenberg text."""
    # Explicitly remove Gutenberg metadata headers/footers
    start_re = r"\*\*\* START OF (.*?) \*\*\*"
    end_re = r"\*\*\* END OF (.*?) \*\*\*"

    start_match = re.search(start_re, text)
    end_match = re.search(end_re, text)

    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]

    # Normalize whitespace (preserve sentence structure)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Split into paragraphs by double newlines (standard Gutenberg formatting)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Filter paragraphs by word length
    valid_paragraphs = [
        p for p in paragraphs
        if min_length <= len(p.split()) <= max_length
    ]

    # If not enough, truncate longer paragraphs to fit
    if len(valid_paragraphs) < num_passages:
        longer_paragraphs = [p for p in paragraphs if len(p.split()) > max_length]
        for lp in longer_paragraphs:
            truncated = ' '.join(lp.split()[:max_length])
            valid_paragraphs.append(truncated)
            if len(valid_paragraphs) >= num_passages:
                break

    # Return a random selection (or all, if fewer available)
    return random.sample(valid_paragraphs, min(num_passages, len(valid_paragraphs)))


def save_passage(passage_data, era, book_id, idx):
    path = f"{BASE_DIR}/{era}/{book_id}_passage_{idx}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(passage_data, f, indent=2)

def collect_passages_per_era(era, book_count, passages_per_book):
    start_year, end_year = ERAS[era]
    books = get_books_by_era(start_year, end_year, book_count)
    all_passages = []
    for book in tqdm(books, desc=f"Processing books ({era})"):
        content = download_book_content(book["formats"])
        if not content:
            continue
        passages = extract_passages(content, num_passages=passages_per_book)
        for idx, passage in enumerate(passages):
            date = extract_date_hints(passage, (start_year, end_year))
            pdata = {
                "passage": passage,
                "book_id": book["id"],
                "title": book["title"],
                "author": book["authors"][0]["name"],
                "era": era,
                "date": str(date),
                "clues": [],
                "rationale": ""
            }
            save_passage(pdata, era, book["id"], idx)
            all_passages.append(pdata)
        time.sleep(0.5)
    return all_passages

def create_train_val_split(passages, val_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(passages)
    split_idx = int(len(passages) * (1 - val_ratio))
    return passages[:split_idx], passages[split_idx:]

def save_json(data, filename):
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main(book_count=100, passages_per_book=100):
    create_directories()
    all_passages = []
    for era in ERAS:
        era_passages = collect_passages_per_era(era, book_count, passages_per_book)
        save_json(era_passages, f"{BASE_DIR}/{era}/passages.json")
        all_passages.extend(era_passages)
    save_json(all_passages, f"{BASE_DIR}/full/all_passages_for_annotation.json")

    train, val = create_train_val_split(all_passages)
    save_json(train, f"{BASE_DIR}/train/passages.json")
    save_json(val, f"{BASE_DIR}/validation/passages.json")

    print(f"Total passages: {len(all_passages)} (Train: {len(train)}, Val: {len(val)})")

if __name__ == "__main__":
    main()
