import os
import re
from collections import defaultdict

FOLDER_PATH = "myFavRaps"

def preprocess(text):
    """Lowercase text, remove punctuation, and split into words."""
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation
    return text.split()  # Split into words

def build_inverted_index():
    """Builds an inverted index from all text files in the folder."""
    index = defaultdict(set)  # Dictionary mapping words to sets of document IDs
    file_map = {}  # Maps document IDs to filenames

    # Loop through all files in the folder
    for doc_id, filename in enumerate(os.listdir(FOLDER_PATH)):
        if filename.endswith(".txt"):  # Process only .txt files
            file_map[doc_id] = filename  # Store file mapping
            with open(os.path.join(FOLDER_PATH, filename), "r", encoding="utf-8") as f:
                words = preprocess(f.read())  # Process text
                for word in words:
                    index[word].add(doc_id)  # Add word occurrence to index

    return index, file_map

def boolean_search(query, index, file_map):
    """Processes AND, OR, and NOT queries and returns matching filenames."""
    query = query.lower().split()  # Convert query to lowercase and split words

    if "and" in query:
        terms = [t for t in query if t not in ["and", "or", "not"]]
        result = index.get(terms[0], set())  # Start with first word's set
        for term in terms[1:]:
            result &= index.get(term, set())  # Intersect sets (AND operation)

    elif "or" in query:
        result = set()
        for term in query:
            if term not in ["and", "or", "not"]:
                result |= index.get(term, set())  # Union sets (OR operation)

    elif "not" in query:
        terms = [t for t in query if t not in ["and", "or", "not"]]
        all_docs = set(file_map.keys())  # Set of all document IDs
        result = all_docs - index.get(terms[0], set())  # Subtract from full set (NOT operation)

    else:
        result = index.get(query[0], set())  # Handle single-word query

    return [file_map[i] for i in result]  # Return filenames

if __name__ == "__main__":
    index, file_map = build_inverted_index()
    query = input("Enter Boolean query (e.g., 'word1 AND word2 NOT word3'): ")
    results = boolean_search(query, index, file_map)

    print("\nMatching files:")
    for file in results:
        print(file)
