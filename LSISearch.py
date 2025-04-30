# LSI Search Engine for your rap lyrics
# Written simply with comments for learning :)

import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style, init
init(autoreset=True)  # Ensures colors reset after each line

# Path to your lyrics folder
FOLDER_PATH = "myFavRaps"  # This is where your rap .txt files are stored

# 1. Preprocessing function to clean up the text
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # remove punctuation and lowercase
    return text

# 2. Load all documents into a list
def load_documents(folder_path):
    docs = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                content = preprocess(file.read())
                docs.append(content)
                filenames.append(filename)
    return docs, filenames

# 3. Build TF-IDF matrix
def build_tfidf_matrix(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer

# 4. Apply LSI using SVD
# n_components=100 means we keep 100 topics (can tune this later)
def apply_lsi(tfidf_matrix, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    lsi_matrix = svd.fit_transform(tfidf_matrix)
    return lsi_matrix, svd

# 5. Search function using LSI or normal TF-IDF
def search(query, tfidf_matrix, documents, vectorizer, filenames, lsi_matrix=None, lsi_model=None):
    query_vec = vectorizer.transform([preprocess(query)])  # preprocess and vectorize query

    if lsi_model is not None:
        query_vec = lsi_model.transform(query_vec)  # project query into LSI space

    # compute cosine similarity between query and all documents
    sims = cosine_similarity(query_vec, lsi_matrix if lsi_model else tfidf_matrix)[0]
    ranked_results = sorted(zip(filenames, sims), key=lambda x: x[1], reverse=True)
    return ranked_results

# 6. Nicely formatted print with colors
def print_results(title, results, comparison_results=None):
    print(f"\n{Style.BRIGHT}{Fore.CYAN}=== {title} ==={Style.RESET_ALL}")
    
    # If we have comparison results, use them to highlight differences
    only_in_these_results = []
    position_changes = {}
    
    if comparison_results:
        comparison_filenames = [filename for filename, _ in comparison_results[:10]]
        these_filenames = [filename for filename, _ in results[:10]]
        
        # Find files only in these results (not in comparison)
        only_in_these_results = [f for f in these_filenames if f not in comparison_filenames]
        
        # Calculate position changes (if a file moved up or down in ranking)
        for i, (filename, _) in enumerate(results[:10]):
            if filename in comparison_filenames:
                comp_pos = comparison_filenames.index(filename)
                if comp_pos != i:
                    position_changes[filename] = comp_pos - i  # positive means moved up in rank
    
    # Print results with appropriate colors
    for i, (filename, score) in enumerate(results[:10]):  # Top 10 results
        if filename in only_in_these_results:
            # Purple for results only visible in this method
            print(f"{Fore.MAGENTA}{Style.BRIGHT}{filename:<60} {Fore.GREEN}Score: {score:.4f} {Fore.MAGENTA}[UNIQUE]")
        elif filename in position_changes:
            change = position_changes[filename]
            # Significantly moved up in ranking (by at least 2 positions)
            if change >= 2:
                print(f"{Fore.BLUE}{filename:<60} {Fore.GREEN}Score: {score:.4f} {Fore.BLUE}[↑ {abs(change)} positions]")
            # Significantly moved down in ranking (by at least 2 positions)
            elif change <= -2:
                print(f"{Fore.RED}{filename:<60} {Fore.GREEN}Score: {score:.4f} {Fore.RED}[↓ {abs(change)} positions]")
            else:
                print(f"{Fore.YELLOW}{filename:<60} {Fore.GREEN}Score: {score:.4f}")
        else:
            print(f"{Fore.YELLOW}{filename:<60} {Fore.GREEN}Score: {score:.4f}")

# === Main program ===
if __name__ == "__main__":
    # Step 1: Load documents
    documents, filenames = load_documents(FOLDER_PATH)

    # Step 2: Build TF-IDF matrix
    tfidf_matrix, vectorizer = build_tfidf_matrix(documents)

    # Step 3: Build LSI model
    lsi_matrix, lsi_model = apply_lsi(tfidf_matrix, n_components=100)

    # Ask user for a search query
    query = input(f"{Style.BRIGHT}{Fore.CYAN}Enter your search query: {Style.RESET_ALL}")

    # Step 4: Search without LSI (TF-IDF)
    normal_results = search(query, tfidf_matrix, documents, vectorizer, filenames)
    print_results("Results WITHOUT LSI (TF-IDF)", normal_results)

    # Step 5: Search with LSI
    lsi_results = search(query, tfidf_matrix, documents, vectorizer, filenames, lsi_matrix, lsi_model)
    print_results("Results WITH LSI (Latent Semantic Indexing)", lsi_results, normal_results)
    
    # Step 6: Show more results from LSI (beyond top 10)
    print(f"\n{Style.BRIGHT}{Fore.CYAN}=== Extended LSI Results (11-20) ==={Style.RESET_ALL}")
    for filename, score in lsi_results[10:20]:  # Results 11-20
        normal_top10 = [f for f, _ in normal_results[:10]]
        if filename not in normal_top10:
            print(f"{Fore.MAGENTA}{Style.BRIGHT}{filename:<60} {Fore.GREEN}Score: {score:.4f} {Fore.MAGENTA}[LSI ONLY]")
        else:
            print(f"{Fore.YELLOW}{filename:<60} {Fore.GREEN}Score: {score:.4f}")

    # Step 7: Explain what the colors mean
    print(f"\n{Style.BRIGHT}{Fore.CYAN}=== Color Legend ==={Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}Purple{Style.RESET_ALL}: Results unique to this method or not in top 10 of other method")
    print(f"{Fore.BLUE}Blue{Style.RESET_ALL}: Results that moved up significantly in ranking")
    print(f"{Fore.RED}Red{Style.RESET_ALL}: Results that moved down significantly in ranking")
    print(f"{Fore.YELLOW}Yellow{Style.RESET_ALL}: Results with similar ranking in both methods")