import requests
import json
import os
import random
import time
import re
import logging
from tqdm import tqdm
from datetime import datetime
import argparse

from search_book_date import get_book_publication_year

### HYPERPARAMETERS ###
# NOTE: ERAS definition is a deliberately contentious choice.
# The idea is to demonstrate how to get an LLM to learn in a way that is culturally relevant to a use case.
# There are overlapping date windows for such historical periods, and many cultures wouldn't use these Euro-centric labels. 
# The idea shows how GRPO-like methods can align an LLM to a "sovereign AI" approach,
# where cultural preferences can be chosen at national-, local-, or business-level
# instead of by a single maximally general LLM provider.
ERAS = {
    "renaissance": (1500, 1650),
    "enlightenment": (1650, 1800),
    "romantic": (1800, 1837),
    "victorian": (1837, 1901),
    "edwardian": (1901, 1920),
    "modern": (1920, 1960)
}
LANGUAGE = "en"
# NOTE: Project Gutenberg is free for any usage under United States law.
# These books are all already public domain. 
# IMPORTANT: If you are based in another country, it is YOUR responsibility to understand what books you can(not) distribute legally.

### CONSTANTS ###
BASE_DIR = "gutenberg_dataset"
LOG_DIR = "logs"

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{BASE_DIR}/{LOG_DIR}/download_log_{timestamp}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            # logging.StreamHandler() 
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}/{LOG_DIR}", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/full", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/train", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/validation", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/test", exist_ok=True)
    for era in ERAS:
        os.makedirs(f"{BASE_DIR}/{era}", exist_ok=True)

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

def load_book_cache():
    cache_path = f"{BASE_DIR}/full/book_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_book_cache(cache):
    cache_path = f"{BASE_DIR}/full/book_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def get_books_for_all_eras(min_books_per_era, logger):
    """Collect books for all eras simultaneously, categorizing as we go."""
    books_by_era = {era: [] for era in ERAS}
    all_books_seen = set()  # Track IDs to avoid duplicates
    page = 1
    
    # Load book cache to avoid repeated lookups
    book_cache = load_book_cache()
    books_found = 0
    books_needed = min_books_per_era * len(ERAS)
    
    # Stats tracking
    stats = {
        "total_books_examined": 0,
        "books_placed": 0,
        "books_outside_range": 0,
        "search_failures": 0,
        "era_stats": {era: 0 for era in ERAS}
    }
    
    # Set up progress tracking for all eras
    with tqdm(total=books_needed, desc="Gathering books for all eras") as pbar:
        while books_found < books_needed and any(len(books_by_era[era]) < min_books_per_era for era in ERAS):
            res = requests.get(f"https://gutendex.com/books/?page={page}").json()
            if not res.get("results"):
                logger.info(f"No more results from Gutendex at page {page}. Stopping collection.")
                break
                
            for book in res["results"]:
                stats["total_books_examined"] += 1
                
                if book["id"] in all_books_seen or LANGUAGE not in book["languages"] or not book["authors"]:
                    continue
                    
                all_books_seen.add(book["id"])
                author = book["authors"][0]
                # if not author["birth_year"] or not author["death_year"]:
                #     continue
                
                # Check cache first
                if str(book["id"]) in book_cache:
                    est_year = int(book_cache[str(book["id"])]["year"])
                    logger.debug(f"Using cached year {est_year} for book: `{book['title']}` id:{book['id']}.")
                else:
                    est_year = get_book_publication_year(book['title'])
                    # Update cache
                    book_cache[str(book["id"])] = {"title": book["title"], "year": est_year}
                    # Save cache every 10 books to prevent data loss
                    if len(book_cache) % 10 == 0:
                        save_book_cache(book_cache)
                
                if est_year is None:
                    logger.info(f'SKIP - search failed for book: `{book["title"]}` id:{book["id"]}.')
                    stats["search_failures"] += 1
                    continue
                
                # Check if this book fits any era
                placed = False
                for era, (start_year, end_year) in ERAS.items():
                    if start_year <= est_year <= end_year:
                        book["estimated_date"] = est_year
                        books_by_era[era].append(book)
                        placed = True
                        books_found += 1
                        pbar.update(1)
                        stats["books_placed"] += 1
                        stats["era_stats"][era] += 1
                        logger.info(f'PLACED - book: `{book["title"]}` id:{book["id"]} in era {era} ({start_year}-{end_year}), year {est_year}.')
                        break
                
                if not placed:
                    # Book didn't fit any era
                    assert est_year < 1500 or est_year > 1960, f"There is a bug. {est_year} is not being mapped to an era."
                    stats["books_outside_range"] += 1
                    logger.info(f'OUTSIDE RANGE - book: `{book["title"]}` id:{book["id"]} publication year {est_year} not in any defined era.')
            
            page += 1
            time.sleep(0.1)
            
            # Check if we've met all our targets
            if all(len(books_by_era[era]) >= min_books_per_era for era in ERAS):
                logger.info("All eras have reached their target book counts. Stopping collection.")
                break
    
    save_book_cache(book_cache)
    
    # Log statistics
    logger.info("=== Book Collection Statistics ===")
    logger.info(f"Total books examined: {stats['total_books_examined']}")
    logger.info(f"Total books placed: {stats['books_placed']}")
    logger.info(f"Books outside range: {stats['books_outside_range']}")
    logger.info(f"Search failures: {stats['search_failures']}")
    logger.info("Books per era:")
    for era in ERAS:
        logger.info(f"  - {era}: {stats['era_stats'][era]}/{min_books_per_era}")
    
    return books_by_era

def extract_passages(text, min_length=200, max_length=400, num_passages=10):
    """Extract `num_passages` coherent passages from a Project Gutenberg text."""
    # Explicitly remove metadata headers/footers in all Project Gutenberg files.
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

    # Return a random selection, or all, if it is a smallish book.
    return random.sample(valid_paragraphs, min(num_passages, len(valid_paragraphs)))

def save_passage(passage_data, era, book_id, idx):
    path = f"{BASE_DIR}/{era}/{book_id}_passage_{idx}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(passage_data, f, indent=2)

def process_era_books(era, books, passages_per_book, logger):
    logger.info(f"Processing {len(books)} books for {era}...")
    era_passages = []
    
    for book in tqdm(books, desc=f"Processing books ({era})"):
        content = download_book_content(book["formats"])
        if not content:
            logger.warning(f"Could not download content for book: {book['title']} id:{book['id']}")
            continue
            
        passages = extract_passages(content, num_passages=passages_per_book)
        logger.info(f"Extracted {len(passages)} passages from book: {book['title']} id:{book['id']}")
        
        for idx, passage in enumerate(passages):
            date = book["estimated_date"]
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
            era_passages.append(pdata)
        time.sleep(0.5)
    
    return era_passages

def create_train_val_split(passages, val_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(passages)
    split_idx = int(len(passages) * (1 - val_ratio))
    return passages[:split_idx], passages[split_idx:]

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main(books_per_era=250, passages_per_book=10):

    create_directories()
    logger = setup_logging()
    logger.info(f"Starting book collection with target of {books_per_era} books per era")
    
    books_by_era = get_books_for_all_eras(books_per_era, logger)
    all_passages = []
    for era, books in books_by_era.items():
        era_passages = process_era_books(era, books, passages_per_book, logger)
        save_json(era_passages, f"{BASE_DIR}/{era}/passages.json")
        all_passages.extend(era_passages)
        logger.info(f"Completed processing for era {era}: {len(era_passages)} passages collected")
    
    # Create train/val/test splits
    save_json(all_passages, f"{BASE_DIR}/full/passages.json")
    train, val = create_train_val_split(all_passages)
    val, test = create_train_val_split(val, val_ratio=0.5)
    save_json(train, f"{BASE_DIR}/train/passages.json")
    save_json(val, f"{BASE_DIR}/validation/passages.json")
    save_json(test, f"{BASE_DIR}/test/passages.json")

    logger.info(f"=== Dataset Creation Complete ===")
    logger.info(f"Total passages: {len(all_passages)}")
    logger.info(f"Train set: {len(train)} passages")
    logger.info(f"Validation set: {len(val)} passages")
    logger.info(f"Test set: {len(test)} passages")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", "--books-per-era", default=100, help="Minimum number of books per era.")
    parser.add_argument('-np', "--passages-per-book", default=250, help='Minimum number of passages per book.')
    args = parser.parse_args()
    main(books_per_era=args.books_per_era, passages_per_book=args.passages_per_book)