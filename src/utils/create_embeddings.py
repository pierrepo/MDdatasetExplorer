"""
Generate Embeddings for Molecular Dynamics Datasets.

This script processes JSON file containing molecular dynamics datasets, and computes embeddings
for the text data using the specified transformer model, and stores the resulting vectors
in the results/embeddings directory. Each dataset is processed individually, and its vectors
are stored in a dedicated Chroma database under results/embeddings/chroma_db_<dataset_name>_<model_name>.

Usage:
======
    python src/utils/create_embeddings.py  --model_name <model_name> --dataset_file_name <dataset_file_name>

Argument:
=========
    --model_name : str
        The name of the transformer model to use for generating embeddings. The model must be
        in ["BERT", "SciBERT", "BioBERT", "SciNCL", "SBERT", "PubMedBERT", "all-MiniLM-L6-v2"].
    --dataset_file_name : str
        The file name of the JSON file with the datasets to process.

Example:
========
    python src/utils/create_embeddings.py --model_name "SciBERT" --dataset_file_name "extended_dataset.json"

This command will generate embeddings for all datasets in the extended_dataset.json file in the results/datasets directory using the SciBERT model.
The resulting embeddings will be stored in the results/embeddings directory in a Chroma database named chroma_db_extended_dataset_Scibert.
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
from typing import List, Dict, Tuple

import torch
import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ProcessPoolExecutor, as_completed


# CONSTANTS
IN_DIR = "results/datasets"
OUT_DIR = "results/embeddings"

SUPPORTED_MODELS = {
    "BERT": "bert-base-uncased",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BioBERT": "dmis-lab/biobert-v1.1",
    "SciNCL": "malteos/scincl",
    "SBERT": "sentence-transformers/all-mpnet-base-v2",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "all-MiniLM-L6-v2" : "sentence-transformers/all-MiniLM-L6-v2"
}


# FUNCTIONS
def get_args() -> str:
    """Parse command-line arguments.

    Returns
    -------
    model_name : str
        The name of the transformer model to use for generating embeddings.
    """
    logger.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="Generate embeddings for datasets using a transformer model.")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=SUPPORTED_MODELS.keys(),
        required=True,
        help=f"The name of the model to use for generating embeddings. Supported models: {', '.join(SUPPORTED_MODELS.keys())}."
    )
    parser.add_argument(
        "--dataset_file_name",
        type=str,
        required=True,
        help="The name of the JSON file containing the datasets to process."
    )
    args = parser.parse_args()

    logger.debug(f"Model name: {args.model_name}")
    logger.debug(f"Dataset file name: {args.dataset_file_name}")
    logger.success("Command-line arguments parsed successfully. \n")

    return args.model_name, args.dataset_file_name


def load_json_file(file_path: str) -> Dict[str, dict]:
    """
    Load a single JSON file and return a dictionary containing the file path and its content.

    Parameters
    ----------
    file_path : str
        The path to the JSON file to load.
    
    Returns
    -------
    Dict[str, dict]
        A dictionary containing the file path and JSON content.
    """
    logger.info(f"Loading JSON file from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_content = json.load(file)
            dataset_count = len(json_content) if isinstance(json_content, list) else 1
            logger.debug(f"File: {file_path} | Number of datasets: {dataset_count}")
            logger.success(f"Loaded JSON file successfully: {file_path} \n")
            return {
                'file_path': file_path,
                'json_content': json_content
            }
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding {file_path}: {e}")
        raise ValueError(f"Failed to load JSON file at {file_path}") from e
   

def load_model(model_path: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a transformer model and its tokenizer from HuggingFace.

    Parameters
    ----------
    model_path : str
        Path to the model on HuggingFace via Transformers or SentenceTransformer.

    Returns
    -------
    model : torch.nn.Module
        The loaded transformer model.
    tokenizer : AutoTokenizer or None
        The tokenizer for the model, or None if using SentenceTransformer.
    """
    logger.info("Loading transformer model from HuggingFace...")
    tokenizer = None

    if model_path.startswith("sentence-transformers"):
        # Load a SentenceTransformer model
        model = SentenceTransformer(model_path)
        vector_dim = model.get_sentence_embedding_dimension()
        logger.success(f"SentenceTransformer model '{model_path}' loaded successfully with vector dimension {vector_dim}. \n")
    else:
        # Load a Transformer model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        vector_dim = model.config.hidden_size
        logger.success(f"HuggingFace model '{model_path}' loaded successfully with vector dimension {vector_dim}. \n")
    
    return model, tokenizer


def clean_and_concatenate(data: dict, excluded_keys=None, words_to_remove=None) -> str:
    """
    Clean and concatenate values from a dictionary, excluding certain keys.

    Parameters
    ----------
    data : dict
        The dictionary containing the data.
    excluded_keys : list, optional
        Keys to exclude from the concatenation (default: ['id', 'origin', 'author', 'date_creation']).
    words_to_remove : list, optional
        Words or phrases to remove from the concatenated string (default: None).

    Returns
    -------
    str
        The cleaned and concatenated string.
    """
    if excluded_keys is None:
        excluded_keys = ['id', 'origin', 'author', 'date_creation']
    
    if words_to_remove is None:
        words_to_remove = ["None", "none", "Unknown", "n/a", "undefined"]

    # Filter out excluded keys and concatenate the remaining values
    concatenated_text = " ".join(
        str(value) for key, value in data.items() 
        if key not in excluded_keys and value
    )

    # Remove unwanted words/phrases
    for word in words_to_remove:
        concatenated_text = concatenated_text.replace(word, "")
    
    return concatenated_text.strip()


def process_batch(batch: List[str], model: torch.nn.Module, tokenizer: AutoTokenizer = None) -> List[List[float]]:
        """Process a single batch to generate embeddings.
        
        Parameters
        ----------
        batch : List[str]
            A list of texts to generate embeddings for.
        model : torch.nn.Module
            The transformer model to use for generating embeddings.
        tokenizer : AutoTokenizer or None
            The tokenizer for the model, or None if using SentenceTransformer.
        """
        embeddings = []
        if tokenizer:  # For Transformers
            with torch.no_grad():
                inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=model.config.max_position_embeddings).to(model.device)
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
                embeddings.extend(batch_embeddings)
        else:  # For SentenceTransformer
            embeddings = model.encode(batch, show_progress_bar=False, device=model.device)
        return embeddings


def generate_embeddings(tokenizer, model, texts, batch_size=16, num_workers=-1) -> List[List[float]]:
    """Generate embeddings for a list of texts using the specified model and tokenizer, with parallel processing.

    Parameters
    ----------
    tokenizer : AutoTokenizer or None
        The tokenizer for the model.
    model : torch.nn.Module
        The embedding model.
    texts : List[str]
        A list of texts to generate embeddings for.
    batch_size : int, optional
        The size of each batch for processing (default is 16).
    num_workers : int, optional
        The number of parallel workers to use (default is -1 for all available cores).

    Returns
    -------
    embeddings : List[List[float]]
        A list of embeddings for each text.
    """
    # Clean the texts
    cleaned_texts = [clean_and_concatenate(json.loads((text))) for text in texts]

    # Split passages into batches
    batches = [cleaned_texts[i:i + batch_size] for i in range(0, len(cleaned_texts), batch_size)]

    # Parallel processing
    embeddings = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_batch, batch, model, tokenizer): batch for batch in batches}
        for future in tqdm(as_completed(futures), total=len(batches), desc="Generating embeddings", ncols=100, colour="blue", unit="batch"):
            try:
                embeddings.extend(future.result())
            except Exception as e:
                print(f"Error processing batch: {e}")

    # Normalize embeddings
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings


def store_in_chroma(vectors: List[List[float]], dataset_json: dict, dataset_name: str, model_name: str) -> None:
    """Store embeddings and corresponding metadata in a Chroma database.
    
    Parameters
    ----------
    vectors : List[List[float]]
        A list of embeddings to store in the Chroma database.
    dataset_json : dict
        The JSON content of the dataset.
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model used to generate the embeddings.    
    """
    # Create the output directory if it does not exist
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    # Initialize a Chroma client
    chroma_db_path = os.path.join(OUT_DIR, f"chroma_db_{dataset_name}_{model_name}")
    client = chromadb.PersistentClient(path=chroma_db_path)
    # Create a new collection with the dataset name and model name
    # Use cosine similarity to search for similar embeddings
    embedding_model = "all-mpnet-base-v2" if model_name != "all-MiniLM-L6-v2"  else "all-MiniLM-L6-v2"
    embdedding_fun = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    db = client.create_collection(name=f"chroma_db_{dataset_name}_{model_name}", metadata={"hnsw:space": "cosine"}, embedding_function=embdedding_fun)

    for idx, vector in tqdm(
        enumerate(vectors),
        desc="Storing embeddings in Chroma",
        total=len(vectors),
        unit="vector",
        ncols=100,
        colour="blue"
    ):
        metadata = dataset_json[idx]
        unique_id = metadata.get("id", idx)
        metadata_str = {key: str(value) for key, value in metadata.items()}
        db.add(
            ids=[unique_id],
            embeddings=vector,
            metadatas=[metadata_str],
            documents=[json.dumps(metadata)]
        )
    logger.success(f"Stored embeddings for {dataset_name} using {model_name} in Chroma successfully.\n")


def process_dataset(dataset: dict, tokenizer, model, model_name: str) -> None:
    """Process a single dataset and generate the corresponding embeddings.
    
    Parameters
    ----------
    dataset : dict
        The dataset to process.
    tokenizer : AutoTokenizer or None
        The tokenizer for the model.
    model : torch.nn.Module
        The embedding model.
    model_name : str
        The name of the model used to generate the embeddings.
    """
    dataset_name = os.path.basename(dataset['file_path']).replace('.json', '')
    logger.info(f"Processing dataset: {dataset_name} with model: {model_name}")

    # Extract documents from the dataset
    documents = [json.dumps(item) for item in dataset['json_content']]

    # Generate embeddings for the documents
    embeddings = generate_embeddings(
            tokenizer=tokenizer,
            model=model,
            texts=documents,
            batch_size=16,
            num_workers=4
        )
    # Store the embeddings in Chroma
    store_in_chroma(embeddings, dataset['json_content'], dataset_name, model_name)


# MAIN PROGRAM
if __name__ == "__main__":
    # Parse command-line arguments
    model_name, dataset_file_name = get_args()
    model_path = SUPPORTED_MODELS[model_name]

    # Load the transformer model
    model, tokenizer = load_model(model_path)
    
    # Load dataset from JSON file
    dataset = load_json_file(f"{IN_DIR}/{dataset_file_name}")

    # Process the dataset and store embeddings in Chroma
    process_dataset(dataset, tokenizer, model, model_name)
