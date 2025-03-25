import requests
import gzip
import io
from io import BytesIO
import csv
from io import StringIO
import re
import logging
import zipfile


class ProjectGutenbergMetadata:

    def __init__(self):
        self.file_url = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
        self.csv_text = ""
        try:
            self.download()
        except Exception as e:
            print(f'Woah\n{e}')

    def download(self):
        r = requests.get(self.file_url)
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
            self.csv_text = f.read().decode("utf-8")

    def sample(self, n=3):
        if not self.csv_text:
            raise ValueError("CSV data is not loaded. Please run download() first.")
        reader = csv.DictReader(StringIO(self.csv_text))
        return [next(reader) for _ in range(n)]

    def search_by_author(self, name):
        if not self.csv_text:
            raise ValueError("CSV data is not loaded. Please run download() first.")
        author_books = [book for book in csv.DictReader(StringIO(self.csv_text)) 
                        if name.lower() in book['Authors'].lower()]
        return author_books

    def search_by_subject(self, name):
        if not self.csv_text:
            raise ValueError("CSV data is not loaded. Please run download() first.")
        subject_books = [book for book in csv.DictReader(StringIO(self.csv_text)) 
                        if name.lower() in book['Subjects'].lower()]
        return subject_books

    def search_by_decade(self, decade):
        if not self.csv_text:
            raise ValueError("CSV data is not loaded. Please run download() first.")
        start_year = (decade // 10) * 10
        end_year = start_year + 9
        decade_books = [book for book in csv.DictReader(StringIO(self.csv_text)) 
                        if start_year <= int(book['Issued'][:4]) <= end_year]
        return decade_books

    def search_by_century(self, century):
        if not self.csv_text:
            raise ValueError("CSV data is not loaded. Please run download() first.")
        start_year = (century - 1) * 100
        end_year = start_year + 99
        century_books = [book for book in csv.DictReader(StringIO(self.csv_text)) 
                         if start_year <= int(book['Issued'][:4]) <= end_year]
        return century_books

    def search_by_year(self, year):
        if not self.csv_text:
            raise ValueError("CSV data is not loaded. Please run download() first.")
        year_books = [book for book in csv.DictReader(StringIO(self.csv_text)) 
                      if int(book['Issued'][:4]) == year]
        return year_books

    def search_by_language(self, language):
        if not self.csv_text:
            raise ValueError("CSV data is not loaded. Please run download() first.")
        language_books = [book for book in csv.DictReader(StringIO(self.csv_text)) 
                          if language.lower() in book['Language'].lower()]
        return language_books

    def search_by_title(self, title):
        if not self.csv_text:
            raise ValueError("CSV data is not loaded. Please run download() first.")
        title_books = [book for book in csv.DictReader(StringIO(self.csv_text)) 
                       if title.lower() in book['Title'].lower()]
        return title_books