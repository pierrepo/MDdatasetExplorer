"""Generate TF-IDF Vectors for Molecular Dynamics Datasets.

This script processes JSON files in the results/datasets directory, computes TF-IDF vectors (Term Frequency-Inverse Document Frequency) for the text data,
and stores the resulting vectors in the results/tfidf_vectors directory. Each dataset is processed individually, and its vectors are stored
in a dedicated Chroma database under results/tfidf_vectors/chroma_db_<dataset_name>.

TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents.
It helps to highlight relevant terms while reducing the weight of common words across all documents.

Usage:
======
    python src/create_tfidf_vectors.py
"""


# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import json
from tqdm import tqdm
from typing import List, Dict

import nltk
import chromadb
from loguru import logger
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


# CONSTANTS
IN_DIR = "results/datasets"
OUT_DIR = "results/tfidf_vectors"


# FUNCTIONS
def load_json_files(directory: str) -> List[Dict[str, dict]]:
    """Load all JSON files from the specified directory and return a list of dictionaries
       containing file path and JSON content for each dataset.
    
    Parameters
    ----------
    directory : str
        The directory containing the JSON files.

    Returns
    -------
    json_data : List[Dict[str, dict]]
        A list of dictionaries where each dictionary contains the file path (`file_path`)
        and the corresponding JSON content (`json_content`).
    """
    logger.info(f"Loading JSON files from {directory}...")
    json_data = []
    counter = 0
    
    for filename in os.listdir(directory):
        # Only process JSON files
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, 'r', encoding='utf-8') as file:
                try:
                    json_content = json.load(file)
                    dataset_count = len(json_content) if isinstance(json_content, list) else 1
                    json_data.append({
                        'file_path': filepath,
                        'json_content': json_content
                    })
                    counter += 1
                    logger.debug(f"{counter}: {filename} | Path: {filepath} | Number of datasets: {dataset_count}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding {filename}: {e}")

    logger.success(f"Loaded {len(json_data)} JSON files successfully. \n")
    
    return json_data


def generate_tfidf_vectors(documents: List[str], vectorizer: TfidfVectorizer) -> List[List[float]]:
    """Generate TF-IDF vectors from a list of documents using the provided vectorizer.

    Parameters
    ----------
    documents : List[str]
        A list of documents to process.
    vectorizer : TfidfVectorizer
        The pre-fitted TF-IDF vectorizer to transform the documents.
    
    Returns
    -------
    tfidf_vectors : List[List[float]]
        A list of normalized TF-IDF vectors for each document.
    """
    # Transform the documents using the provided vectorizer
    tfidf_matrix = vectorizer.transform(documents)
    
    # Normalize the TF-IDF vectors (to have a norm of 1)
    normalized_tfidf = normalize(tfidf_matrix, norm='l2', axis=1)
    
    # Convert the sparse matrix to a dense array and return as list of lists
    return normalized_tfidf.toarray()


def store_in_chroma(vectors: List[List[float]], dataset_json: dict, dataset_name: str) -> None:
    """Store TF-IDF vectors and corresponding metadata in a Chroma database.
    
    Parameters
    ----------
    vectors : List[List[float]]
        The TF-IDF vectors to store.
    dataset_json : dict
        The metadata associated with the dataset.
    dataset_name : str
        The name of the dataset.
    """
    # Initialize the Chroma client and database
    chroma_db_path = os.path.join(OUT_DIR, f"chroma_db_{dataset_name}")
    client = chromadb.PersistentClient(path=chroma_db_path)
    # Create a new collection for the dataset
    # Use cosine similarity for nearest neighbors
    db = client.create_collection(name=f"chroma_db_{dataset_name}", metadata={"hnsw:space": "cosine"})

    for idx, vector in tqdm(
        enumerate(vectors), 
        desc=f"Storing TF-IDF vectors in Chroma", 
        total=len(vectors),
        unit="vector",
        ncols=100,
        colour="blue"
    ):
        metadata = dataset_json[idx]
        # Convert metadata values to strings
        metadata_str = {key: str(value) if not isinstance(value, list) else ', '.join(str(v) for v in value) for key, value in metadata.items()}
        # Store the vector and metadata in Chroma
        db.add(
            ids=[f"{dataset_name}_{idx}"],  # Unique ID for each vector
            embeddings=vector,  # The actual TF-IDF vector
            metadatas=[metadata_str],  # The metadata associated with this vector
            documents=[json.dumps(metadata)]  # Store the document as a JSON string
        )

    logger.success(f"Stored TF-IDF vectors for {dataset_name} in Chroma successfully.\n")


def process_dataset(dataset: dict, vectorizer: TfidfVectorizer) -> None:
    """Process a single dataset and generate the corresponding TF-IDF vectors.
    
    Parameters
    ----------
    dataset : dict
        The dataset to process. It contains the file path (`file_path`) and the JSON content (`json_content`).
    vectorizer : TfidfVectorizer
        The pre-fitted TF-IDF vectorizer to transform the dataset's documents.
    """
    dataset_name = os.path.basename(dataset['file_path']).replace('.json', '')
    logger.info(f"Processing dataset: {dataset_name}")

    # Extract documents and metadata from the dataset's JSON content
    documents = [json.dumps(item) for item in dataset['json_content']]  # Convert each document to JSON string
    
    # Generate TF-IDF vectors (ensure they are normalized)
    logger.debug("Generating TF-IDF vectors...")
    vectors = []

    for document in tqdm(
        documents, 
        desc=f"Processing documents in {dataset_name}", 
        unit="document", 
        total=len(documents),
        colour="blue",
        ncols=100
    ):
        vectors.append(generate_tfidf_vectors([document], vectorizer)[0])
    
    # Store vectors and metadata in Chroma
    logger.debug("Storing TF-IDF vectors in Chroma...")
    store_in_chroma(vectors, dataset['json_content'], dataset_name)


# MAIN PROGRAM
if __name__ == "__main__":
    # Create output directory if it does not exist
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Necessary for tokenization
    nltk.download('punkt')

    # Load all datasets from the input directory
    datasets = load_json_files(IN_DIR)

    # Combine all datasets to create a global vocabulary
    all_documents = [json.dumps(item) for dataset in datasets for item in dataset['json_content']]

    # Initialize and fit the TF-IDF vectorizer on all documents
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_documents)

    # Process each dataset
    for dataset in datasets:
        process_dataset(dataset, vectorizer)

    logger.success(f" --- All datasets processed successfully. ---")

