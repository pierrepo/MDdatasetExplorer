"""Create TF-IDF Vectors for Molecular Dynamics Datasets

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents.
It helps to highlight relevant terms while reducing the weight of common words across all documents.

Usage:
======
    python src/utils/create_tfidf_vectors.py --dataset_file_name <dataset_file_name>

Argument:
=========
    --dataset_file_name : str
        The file name of the JSON file with the datasets to process.

Example:
========
    python src/utils/create_tfidf_vectors.py --dataset_file_name "extended_dataset.json"

This command will generate TF-IDF vectors for all datasets in the `extended_dataset.json` file in the `results/datasets` directory.
The resulting vectors will be stored in the `results/tfidf_vectors` directory in a chroma database named `chroma_db_tfidf_vectors_<dataset_name>`.
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
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
import chromadb
from chromadb.utils import embedding_functions
from concurrent.futures import ProcessPoolExecutor, as_completed


# CONSTANTS
IN_DIR = "results/datasets"
OUT_DIR = "results/tfidf_vectors"


# FUNCTIONS
def get_args() -> str:
    """Parse command-line arguments."""
    logger.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(
        description="Generate TF-IDF vectors for datasets."
    )
    parser.add_argument(
        "--dataset_file_name",
        type=str,
        required=True,
        help="The name of the JSON file containing the datasets to process.",
    )
    args = parser.parse_args()
    logger.debug(f"Dataset file name: {args.dataset_file_name}")
    logger.success("Command-line arguments parsed successfully. \n")
    return args.dataset_file_name


def load_json_file(file_path: str) -> dict:
    """
    Load a JSON file and return its content.

    Parameters
    ----------
    file_path : str
        The path to the JSON file to load.

    Returns
    -------
    dict
        The content of the JSON file.
    """
    logger.info(f"Loading JSON file from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json_content = json.load(file)
            dataset_count = len(json_content) if isinstance(json_content, list) else 1
            logger.debug(f"File: {file_path} | Number of datasets: {dataset_count}")
            logger.success(f"Loaded JSON file successfully: {file_path} \n")
            return json_content
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding {file_path}: {e}")
        raise ValueError(f"Failed to load JSON file at {file_path}") from e


def clean_and_concatenate(data: dict, excluded_keys=None, words_to_remove=None) -> str:
    """
    Clean and concatenate values from a dictionary, excluding certain keys.

    Parameters
    ----------
    data : dict
        The dictionary containing the data.
    excluded_keys : list, optional
        Keys to exclude from the concatenation.
    words_to_remove : list, optional
        Words or phrases to remove from the concatenated string.

    Returns
    -------
    str
        The cleaned and concatenated string.
    """
    if excluded_keys is None:
        excluded_keys = [
            "id",
            "origin",
            "author",
            "date_creation",
            "url",
            "dt",
            "nsteps",
            "temperature",
            "thermostat",
            "barostat",
        ]

    if words_to_remove is None:
        words_to_remove = ["None", "none", "Unknown", "n/a", "undefined"]

    concatenated_text_parts = []

    for key, value in data.items():
        if key in excluded_keys:
            continue

        if key.startswith("has") and value is True:
            concatenated_text_parts.append(key)
        elif key == "atom_number":
            concatenated_text_parts.append(f"{key}={value}")
        elif isinstance(value, list):
            concatenated_text_parts.append(" ".join(str(item) for item in value))
        elif value:
            concatenated_text_parts.append(str(value))

    concatenated_text = " ".join(concatenated_text_parts)

    for word in words_to_remove:
        concatenated_text = concatenated_text.replace(word, "")

    return concatenated_text.strip()


def process_batch(batch: list, vectorizer: TfidfVectorizer) -> np.ndarray:
    """
    Generate TF-IDF vectors for a batch of text data.

    Parameters
    ----------
    batch : list
        A list of cleaned text strings.
    vectorizer : TfidfVectorizer
        The TF-IDF vectorizer.

    Returns
    -------
    np.ndarray
        The TF-IDF vectors for the batch.
    """
    return vectorizer.transform(batch).toarray()


def store_in_chroma(vectors: np.ndarray, dataset_json: dict, dataset_name: str) -> None:
    """
    Store TF-IDF vectors and corresponding metadata in a Chroma database.

    Parameters
    ----------
    vectors : np.ndarray
        A numpy array of TF-IDF vectors.
    dataset_json : dict
        The JSON content of the dataset.
    dataset_name : str
        The name of the dataset.
    """
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    chroma_db_path = os.path.join(OUT_DIR, f"chroma_db_{dataset_name}_tfidf")
    client = chromadb.PersistentClient(path=chroma_db_path)

    db = client.create_collection(
        name=f"chroma_db_{dataset_name}_tfidf",
        metadata={"hnsw:space": "cosine"},
    )

    for idx, vector in tqdm(
        enumerate(vectors),
        desc="Storing TF-IDF vectors in Chroma",
        total=len(vectors),
        unit="vector",
        ncols=100,
        colour="green",
    ):
        metadata = dataset_json[idx]
        unique_id = metadata.get("id", idx)
        metadata_str = {key: str(value) for key, value in metadata.items()}
        db.add(
            ids=[unique_id],
            embeddings=[vector.tolist()],
            metadatas=[metadata_str],
            documents=[json.dumps(metadata)],
        )
    logger.success(
        f"Stored TF-IDF vectors for {dataset_name} in Chroma successfully.\n"
    )


def process_dataset(dataset: dict, vectorizer: TfidfVectorizer) -> None:
    """
    Process a single dataset and generate the corresponding TF-IDF vectors.

    Parameters
    ----------
    dataset : dict
        The dataset to process.
    vectorizer : TfidfVectorizer
        The TF-IDF vectorizer.
    """
    dataset_name = os.path.basename(dataset["file_path"]).replace(".json", "")
    logger.info(f"Processing dataset: {dataset_name}")

    documents = [clean_and_concatenate(item) for item in dataset["json_content"]]

    vectors = process_batch(documents, vectorizer)

    store_in_chroma(vectors, dataset["json_content"], dataset_name)


# MAIN PROGRAM
if __name__ == "__main__":
    dataset_file_name = get_args()

    logger.info("Initializing TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

    dataset_path = os.path.join(IN_DIR, dataset_file_name)
    dataset_json = load_json_file(dataset_path)

    logger.info("Fitting TF-IDF vectorizer...")
    all_texts = [clean_and_concatenate(item) for item in dataset_json]
    vectorizer.fit(all_texts)

    dataset = {"file_path": dataset_path, "json_content": dataset_json}

    process_dataset(dataset, vectorizer)
