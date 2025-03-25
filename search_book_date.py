import os
import re
import requests

system_prompt = """The user will give you the title of a book, 
run a search and only respond with exactly the number of the year it was written.

Here are a few examples:

User: Find the publication year for the book Pride and Prejudice.
Your response: 1813

User: Find the publication year for the book Romeo and Juliet.
Your response: 1597
"""

def get_book_publication_year(book_name):
    """
    Query the Perplexity API to retrieve the publication year for a given book.
    
    Parameters:
        book_name (str): The title of the book.
        
    Returns:
        int or None: The publication year if found, otherwise None.
    """
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Find the publication year for the book '{book_name}'."
            }
        ],
        "web_search_options": {"search_context_size": "low"}
    }
    headers = {
        "Authorization": f"Bearer {os.environ['PPLX_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    # Parse the response JSON. Adjust the key based on the actual API response structure.
    data = response.json()
    answer = data['choices'][0]['message']['content']
    
    # Extract a 4-digit year from the answer.
    match = re.search(r'\b(1[5-9]\d{2}|20\d{2})\b', answer)
    if match:
        return int(match.group(1))
    else:
        print(data)
    return None

# Example usage:
if __name__ == "__main__":
    book = "Pride and Prejudice"
    year = get_book_publication_year(book)
    if year:
        print(f"The publication year for '{book}' is {year}.")
    else:
        print(f"Publication year for '{book}' could not be determined.")
