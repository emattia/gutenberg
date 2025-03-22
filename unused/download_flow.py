from metaflow import FlowSpec, step, Parameter, current
import requests
import json
import os
import random
import time
import re
from tqdm import tqdm
from collections import defaultdict
import math

### CONSTANTS ###
eras = {
    "renaissance": (1500, 1650),
    "enlightenment": (1650, 1800),
    "victorian": (1837, 1901),
    "edwardian": (1901, 1920),
    "modern": (1920, 1960)
}

class DownloadGutenberg(FlowSpec):
    """
    A flow to download text passages from Project Gutenberg and organize them by historical era.
    Processing is parallelized based on configurable batch sizes.
    """
    
    books_per_era = Parameter('books-per-era', 
                             default=25, 
                             help='Number of books to download per era')
    
    passages_per_book = Parameter('passages-per-book', 
                                 default=20, 
                                 help='Number of passages to extract per book')
    
    min_passage_len = Parameter('min-passage-len', 
                               default=200, 
                               help='Minimum passage length in words')
    
    max_passage_len = Parameter('max-passage-len', 
                               default=400, 
                               help='Maximum passage length in words')
    
    output_dir = Parameter('output-dir', 
                          default='gutenberg_dataset', 
                          help='Directory to store the dataset on remote workers')
    
    books_per_batch = Parameter('books-per-batch', 
                              default=5, 
                              help='Number of books to process in each parallel worker')

    @step
    def start(self):
        """
        Start the flow and create batches of work based on the books_per_batch parameter.
        """
        print(f"Starting Gutenberg dataset download with {self.books_per_era} books per era")
        print(f"Using {self.books_per_batch} books per batch for parallelization")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create era directories
        for era in eras:
            os.makedirs(os.path.join(self.output_dir, era), exist_ok=True)
        
        # First, get all books for all eras without downloading content
        self.all_era_books = {}
        for era, (start_year, end_year) in eras.items():
            print(f"Finding books for {era} era ({start_year}-{end_year})...")
            era_books = self._get_books_by_era(start_year, end_year, era, max_books=self.books_per_era)
            if era_books:
                self.all_era_books[era] = era_books
        
        # Now create work batches across eras
        work_batches = []
        for era, books in self.all_era_books.items():
            # Split books into batches
            for i in range(0, len(books), self.books_per_batch):
                batch_books = books[i:i + self.books_per_batch]
                work_batches.append({
                    'era': era,
                    'books': batch_books,
                    'era_range': eras[era]
                })
        
        self.work_batches = work_batches
        print(f"Created {len(work_batches)} work batches across {len(self.all_era_books)} eras")
        
        self.next(self.download, foreach='work_batches')

    @step
    def download(self):
        """
        Download and process a batch of books from a specific era.
        """
        era = self.input['era']
        books = self.input['books']
        start_year, end_year = self.input['era_range']
        
        print(f"Worker {current.task_id} processing {len(books)} books from {era} era")
        
        self.passages = []
        self.books_processed = 0
        
        # Process each book in this batch
        for book in books:
            print(f"Downloading {book['title']} by {book['authors'][0]['name']}")
            content = self._download_book_content(book["formats"])
            
            if content:
                self.books_processed += 1
                
                # Save full book
                book_filename = os.path.join(self.output_dir, era, f"{book['id']}_full.txt")
                with open(book_filename, "w", encoding="utf-8") as f:
                    f.write(content)
                
                # Extract passages
                passages = self._extract_passages(
                    content, 
                    min_length=self.min_passage_len,
                    max_length=self.max_passage_len,
                    num_passages=self.passages_per_book
                )
                
                # Process and save each passage
                for j, passage in enumerate(passages):
                    passage_date = self._extract_date_hints(passage, (start_year, end_year))
                    if not passage_date:
                        passage_date = book.get("estimated_date", (start_year + end_year) // 2)
                    
                    # Format as string
                    date_str = str(passage_date)
                    
                    # Create passage data
                    passage_data = {
                        "passage": passage,
                        "book_id": book["id"],
                        "title": book["title"],
                        "author": book["authors"][0]["name"],
                        "era": era,
                        "date": date_str,
                        "clues": [],  
                        "rationale": ""
                    }
                    
                    # Save individual passage file
                    passage_filename = os.path.join(self.output_dir, era, f"{book['id']}_passage_{j}.json")
                    with open(passage_filename, "w", encoding="utf-8") as f:
                        json.dump(passage_data, f, indent=2)
                    
                    # Add to collected passages
                    self.passages.append(passage_data)
            
            time.sleep(0.25)  # Be gentle to the API
        
        self.era = era
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Combine results from parallel workers and create the final dataset.
        """
        # Combine passages from all workers
        self.all_passages = []
        for inp in inputs:
            self.all_passages.extend(inp.passages)
        
        print(f"Combined {len(self.all_passages)} passages from {len(inputs)} workers")
        
        # Collect stats
        self.total_books = sum(inp.books_processed for inp in inputs)
        self.passages_per_era = defaultdict(int)
        
        for passage in self.all_passages:
            self.passages_per_era[passage['era']] += 1
        
        # Save combined dataset
        with open(os.path.join(self.output_dir, "all_passages_for_annotation.json"), "w") as f:
            json.dump(self.all_passages, f, indent=2)
        
        # Create combined per-era files
        for era in eras:
            era_passages = [p for p in self.all_passages if p["era"] == era]
            if era_passages:
                with open(os.path.join(self.output_dir, f"{era}_passages.json"), "w") as f:
                    json.dump(era_passages, f, indent=2)
        
        self.next(self.end)

    @step
    def end(self):
        """
        Print summary statistics and complete the flow.
        """
        print("\n=== Download Summary ===")
        print(f"Total books downloaded: {self.total_books}")
        print(f"Total passages created: {len(self.all_passages)}")
        print("\nPassages per era:")
        for era, count in self.passages_per_era.items():
            print(f"  {era}: {count} passages")
        print(f"\nDataset saved to {self.output_dir}/all_passages_for_annotation.json")

    # Helper methods from your original script
    def _extract_date_hints(self, text, era_range):
        """Extract potential dates from text"""
        start_year, end_year = era_range
        
        # Look for explicit years in the range
        year_pattern = r'\b(1[5-9]\d\d|20[0-5]\d)\b'
        years = re.findall(year_pattern, text)
        years = [int(y) for y in years if start_year <= int(y) <= end_year]
        
        # If we found specific years, use the median
        if years:
            return int(sorted(years)[len(years)//2])
        
        # Otherwise, use middle of era range
        return (start_year + end_year) // 2

    def _download_book_content(self, formats):
        """Download a book's content from Gutenberg"""
        # Try formats in preference order
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

    def _get_books_by_era(self, start_year, end_year, era_name, max_books=25):
        """Get books from a specific era using Gutendex API"""
        books = []
        page = 1
        
        while len(books) < max_books and page < 15:
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
                        print(f"Found book: {book['title']} by {author['name']} (est. {estimated_writing_year})")
                        
                        if len(books) >= max_books:
                            break
                
                page += 1
                time.sleep(0.25)  # Be nice to the API
                
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                time.sleep(2)  # Wait longer on error
                
        return books

    def _extract_passages(self, text, min_length=200, max_length=400, num_passages=20):
        """Extract clean passages from text"""
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
        
        # If we don't have enough, take longer ones and truncate
        if len(valid_paragraphs) < num_passages:
            longer_paragraphs = [p for p in paragraphs if len(p.split()) > max_length]
            for p in longer_paragraphs:
                words = p.split()
                valid_paragraphs.append(' '.join(words[:max_length]))
                if len(valid_paragraphs) >= num_passages:
                    break
        
        # Return random selection
        if len(valid_paragraphs) <= num_passages:
            return valid_paragraphs
        return random.sample(valid_paragraphs, num_passages)


if __name__ == '__main__':
    DownloadGutenberg()