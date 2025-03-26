import os
import re
import math
from collections import defaultdict, Counter

FOLDER_PATH = "myFavRaps"

def preprocess(text):
    """Lowercase, remove punctuation, and split into words."""
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation
    return text.split()  # Split into words

def compute_tf(doc):
    """Compute Term Frequency (TF) for a document."""
    word_counts = Counter(doc)  # Count word occurrences
    return {word: count / len(doc) for word, count in word_counts.items()}  # Normalize by document length

def compute_idf(documents):
    """Compute Inverse Document Frequency (IDF) for all terms."""
    N = len(documents)  # Total number of documents
    idf = defaultdict(lambda: 0)

    # Count documents containing each word
    for doc in documents:
        for word in set(doc):  # Unique words only
            idf[word] += 1

    # Compute IDF score
    return {word: math.log(N / freq) for word, freq in idf.items()}

def build_tfidf():
    """Compute TF-IDF vectors for all documents."""
    documents, filenames = [], []
    
    # Read all text files
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(FOLDER_PATH, filename), "r", encoding="utf-8") as f:
                words = preprocess(f.read())  # Process text
                documents.append(words)
                filenames.append(filename)

    idf = compute_idf(documents)  # Compute IDF for entire corpus
    tfidf_vectors = [compute_tf(doc) for doc in documents]  # Compute TF for each document

    # Multiply TF by IDF
    for vector in tfidf_vectors:
        for word in vector:
            vector[word] *= idf[word]

    return filenames, tfidf_vectors, idf

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two TF-IDF vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())  # Common words
    numerator = sum(vec1[word] * vec2[word] for word in intersection)  # Dot product

    # Compute vector magnitudes
    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    return numerator / denominator if denominator != 0 else 0  # Return similarity score

def vector_search(query, filenames, tfidf_vectors, idf):
    """Rank documents based on cosine similarity with the query."""
    query_words = preprocess(query)  # Process the query
    query_vector = compute_tf(query_words)  # Compute TF for query

    # Multiply query TF by IDF
    for word in query_vector:
        if word in idf:
            query_vector[word] *= idf[word]

    # Compute similarity with each document
    scores = [(filenames[i], cosine_similarity(query_vector, tfidf_vectors[i])) for i in range(len(filenames))]
    scores.sort(key=lambda x: x[1], reverse=True)  # Sort results by highest similarity

    return [file for file, score in scores if score > 0]  # Return ranked files

if __name__ == "__main__":
    filenames, tfidf_vectors, idf = build_tfidf()
    query = input("Enter search query: ")
    results = vector_search(query, filenames, tfidf_vectors, idf)

    print("\nRanked results:")
    for file in results:
        print(file)
