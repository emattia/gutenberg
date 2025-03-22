import requests
import json
import os
import random
import time
from datetime import datetime
import re
from tqdm import tqdm

### CONSTANTS ###
eras = {
    "renaissance": (1500, 1650),
    "enlightenment": (1650, 1800),
    "victorian": (1837, 1901),
    "edwardian": (1901, 1920),
    "modern": (1920, 1960)
}
### HYPERPARAMETERS ### 
default_n_books_per_era = 25
default_n_passages_per_book = 20
default_min_passage_len = 200
default_max_passage_len = 400

# Create output directories
base_dir = "gutenberg_dataset"
os.makedirs(base_dir, exist_ok=True)
for era in eras:
    os.makedirs(f"{base_dir}/{era}", exist_ok=True)

# Function to extract date hints from text
def extract_date_hints(text, era_range):
    """Extract potential dates or time period indicators from the text"""
    start_year, end_year = era_range
    
    # Look for explicit years in the range
    year_pattern = r'\b(1[5-9]\d\d|20[0-5]\d)\b'
    years = re.findall(year_pattern, text)
    years = [int(y) for y in years if start_year <= int(y) <= end_year]
    
    # If we found specific years, use the median of those years
    if years:
        return int(sorted(years)[len(years)//2])
    
    # Otherwise, use middle of era range
    return (start_year + end_year) // 2

# Function to download a book's content
def download_book_content(formats):
    # Try to get plain text first (in preference order)
    for format_type in [
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
        "text/html"
    ]:
        if format_type in formats:
            url = formats[format_type]
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return response.text
            except Exception as e:
                print(f"Error downloading {format_type}: {e}")
                continue
    
    return None

# Function to get books from a specific time period with improved author-based dating
def get_books_by_era(start_year, end_year, era_name, max_books=100):
    books = []
    page = 1

    pbar = tqdm(total=max_books, desc=f"Discovering books in {era_name} era...")
    
    while len(books) < max_books and page < 15:  # Increased page limit for more results
        try:
            # Query Gutendex API
            response = requests.get(f"https://gutendex.com/books/?page={page}")
            data = response.json()
            
            if not data.get("results"):
                break
                
            for book in data["results"]:

                # Only process books with author information
                if not book["authors"]:
                    continue

                # Only process books in english.
                if "en" not in book["languages"]:
                    continue
                    
                author = book["authors"][0]
                
                # Skip if we don't have birth/death data
                if not author["birth_year"] or not author["death_year"]:
                    continue
                
                # Estimate book's writing date (when author was ~40 years old)
                author_prime_age = 40
                estimated_writing_year = min(
                    author["birth_year"] + author_prime_age,
                    author["death_year"]
                )
                
                # Check if estimated year is in our desired era
                if start_year <= estimated_writing_year <= end_year:
                    # Add estimated date to the book info
                    book["estimated_date"] = estimated_writing_year
                    books.append(book)
                    pbar.set_description(f"Found book: {book['title']} by {author['name']} (est. {estimated_writing_year})")
                    
                    if len(books) >= max_books:
                        break
            
            page += 1
            time.sleep(0.25) # Be gentle to the API
            
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            time.sleep(2) # Wait longer on error

        pbar.update(1)

    pbar.close()
            
    return books

# Function to extract clean passages
def extract_passages(text, min_length=default_min_passage_len, max_length=default_max_passage_len, num_passages=default_n_passages_per_book):
    """Extract several passages from text, ensuring they're clean and coherent"""
    # Remove Gutenberg header/footer
    for marker in ["*** START OF", "*** END OF", "***START OF", "***END OF"]:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                text = parts[1]
    
    # Clean up text
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\.\s+[A-Z]', text) if p.strip()]
    
    # Filter by length
    valid_paragraphs = [p for p in paragraphs if len(p.split()) >= min_length and len(p.split()) <= max_length]
    
    # If we don't have enough paragraphs, take longer ones and truncate
    if len(valid_paragraphs) < num_passages:
        longer_paragraphs = [p for p in paragraphs if len(p.split()) > max_length]
        for p in longer_paragraphs:
            words = p.split()
            valid_paragraphs.append(' '.join(words[:max_length]))
            if len(valid_paragraphs) >= num_passages:
                break
    
    # Return random selection of passages
    if len(valid_paragraphs) <= num_passages:
        return valid_paragraphs
    return random.sample(valid_paragraphs, num_passages)


# Download books for each era
all_books = {}
for era, (start_year, end_year) in eras.items():
    print(f"\nGetting books for {era} era ({start_year}-{end_year})...")
    era_books = get_books_by_era(start_year, end_year, era_name=era, max_books=default_n_books_per_era)
    
    if not era_books:
        print(f"No books found for {era} era. Trying backup method...")
        # Try again with a broader search - could implement a fallback here
        continue
        
    all_books[era] = era_books
    
    # Download and save book content
    pbar = tqdm(total=len(era_books), desc="Downloading book content...")
    for i, book in enumerate(era_books):
        pbar.set_description(f"Downloading {book['title']}...")
        content = download_book_content(book["formats"])
        
        if content:

            # Save full book
            book_filename = f"{base_dir}/{era}/{book['id']}_full.txt"
            with open(book_filename, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Derive from passages from full text
            passages = extract_passages(content, num_passages=default_n_passages_per_book)
            
            for j, passage in enumerate(passages):
                passage_date = extract_date_hints(passage, (start_year, end_year))
                if not passage_date:
                    passage_date = book.get("estimated_date", (start_year + end_year) // 2)
                
                # Format the date as a string (year only)
                date_str = str(passage_date)
                
                # Save passage with metadata
                passage_data = {
                    "passage": passage,
                    "book_id": book["id"],
                    "title": book["title"],
                    "author": book["authors"][0]["name"],
                    "era": era,
                    "date": date_str, 

                    ### To be filled during annotation and/or LLM reasoning ###
                    "clues": [],  
                    "rationale": ""
                }
                
                passage_filename = f"{base_dir}/{era}/{book['id']}_passage_{j}.json"
                with open(passage_filename, "w", encoding="utf-8") as f:
                    json.dump(passage_data, f, indent=2)
            
        pbar.update()
        time.sleep(0.25)  # Be gentle to the API

# Create a merged dataset file for annotation
all_passages = []
for era_dir in os.listdir(base_dir):
    era_path = os.path.join(base_dir, era_dir)
    if os.path.isdir(era_path):
        for file in os.listdir(era_path):
            if "_passage_" in file and file.endswith(".json"):
                with open(os.path.join(era_path, file), "r", encoding="utf-8") as f:
                    passage_data = json.load(f)
                    # Ensure date field exists
                    if "date" not in passage_data:
                        era_range = eras.get(passage_data["era"], (1800, 1900))
                        passage_data["date"] = str((era_range[0] + era_range[1]) // 2)
                    all_passages.append(passage_data)

# Save all passages to a single file for easier annotation
with open(f"{base_dir}/all_passages_for_annotation.json", "w", encoding="utf-8") as f:
    json.dump(all_passages, f, indent=2)

# Also save by era for easier management
for era in eras:
    era_passages = [p for p in all_passages if p["era"] == era]
    with open(f"{base_dir}/{era}_passages.json", "w", encoding="utf-8") as f:
        json.dump(era_passages, f, indent=2)

print(f"\nDownloaded content for {sum(len(books) for books in all_books.values())} books")
print(f"Created {len(all_passages)} passages for annotation")
print(f"Annotation file saved to {base_dir}/all_passages_for_annotation.json")