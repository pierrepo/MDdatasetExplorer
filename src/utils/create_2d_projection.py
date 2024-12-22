"""
Script to Generate 2D Projections of Embeddings for Visualization

This script reads embeddings stored in a Chroma database, reduces their dimensionality 
to two dimensions using a specified method (UMAP or t-SNE), and saves the resulting 
2D coordinates to a .npy file for further visualization or analysis.

Usage
=====
    python src/utils/create_2d_projection.py --db-path <path_to_db> --method <method>

Arguments
=========
    --db-path: str
        Path to the Chroma database containing the embeddings to visualize.
    --method: str
        Dimensionality reduction method to use. Must be one of 'umap' or 'tsne'.

Example
=======
    python src/utils/create_2d_projection.py --db-path results/embeddings/chroma_db_extended_dataset_all-MiniLM-L6-v2 --method umap

This command will read the embeddings stored in the Chroma database located at "results/embeddings/chroma_db_extended_dataset_all-MiniLM-L6-v2", 
reduce their dimensionality to two dimensions using UMAP, and save the resulting 2D coordinates to "results/2d_projections/chroma_db_extended_all-MiniLM-L6-v2_umap.npy".
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import argparse
import numpy as np

import umap
import chromadb
from typing import Tuple
from sklearn.manifold import TSNE
from loguru import logger

# CONSTANTS
OUT_DIR = "results/2d_projections"
METHODS = ["umap", "tsne"]
N_COMPONENTS = 2


# FUNCTIONS
def get_args() -> Tuple[str, str]:
    """Parse command-line arguments.

    Returns
    -------
    Tuple[str, str]
        - db_path: Path to the Chroma database containing the embeddings to visualize.
        - method: Dimensionality reduction method to use.
    """
    logger.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="Generate 2D projections of embeddings for visualization.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to the Chroma database containing the embeddings to visualize.")
    parser.add_argument("--method", type=str, required=True, choices=METHODS, help="Dimensionality reduction method to use.")
    args = parser.parse_args()

    logger.debug(f"db-path: {args.db_path}")
    logger.debug(f"method: {args.method}")
    logger.success("Command-line arguments parsed successfully. \n")

    return args.db_path, args.method


def load_embeddings(db_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and IDs from a Chroma database.

    Parameters
    ----------
    db_path : str
        Path to the Chroma database.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - embeddings: Array of embeddings.
        - labels: Array of labels/IDs.
    """
    logger.info("Loading embeddings from the Chroma database...")
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    if not collections:
        raise ValueError(f"No collections found in the database at {db_path}.")

    collection_name = collections[0].name
    db = client.get_collection(name=collection_name)

    data = db.get(include=["embeddings", "metadatas"])
    embeddings = data.get("embeddings", None)
    labels = data.get("ids", None)
    labels_idx = [label.split("_")[0] for label in labels]
    if embeddings is None:
        raise ValueError(f"No embeddings found in the collection '{collection_name}'.")

    logger.success(f"{len(embeddings)} embeddings loaded successfully. \n")
    return np.array(embeddings), np.array(labels_idx)


def reduce_dimensionality(embeddings: np.array, method: str ="umap") -> np.ndarray:
    """
    Reduce dimensionality of embeddings to 2D using the specified method.

    Parameters
    ----------
    embeddings : np.ndarray
        High-dimensional embeddings.
    method : str
        Dimensionality reduction method ('umap' or 'tsne').

    Returns
    -------
    np.ndarray
        2D coordinates of the embeddings.
    """
    logger.info(f"Reducing dimensionality of embeddings to 2D using {method}...")
    
    # Prepare the reducer based on the chosen method
    if method == "umap":
        reducer = umap.UMAP(n_components=N_COMPONENTS)
    elif method == "tsne":
        reducer = TSNE(n_components=N_COMPONENTS, init="random", perplexity=30)
    else:
        raise ValueError("Invalid method. Choose either 'umap' or 'tsne'.")

    return reducer.fit_transform(embeddings)


def save_2d_coordinates(coordinates: np.ndarray, idx: np.array, db_path: str, method: str) -> None:
    """
    Save the 2D coordinates to a .npy file.

    Parameters
    ----------
    coordinates : np.ndarray
        2D coordinates of the embeddings.
    idx : np.array
        Array of IDs.
    db_path : str
        Path to the Chroma database.
    method : str
        Dimensionality reduction method used ('umap' or 'tsne').
    """
    logger.info("Saving 2D coordinates to a .npy file...")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    out_file = os.path.join(OUT_DIR, f"{os.path.basename(db_path)}_{method}.npy")
    # Save 2D coordinates and IDs to a .npy file
    data = np.hstack((idx.reshape(-1, 1), coordinates))
    np.save(out_file, data)
    logger.success(f"2D coordinates saved to {out_file} successfully. \n")


# MAIN PROGRAM
if __name__ == "__main__":
    # Parse command-line arguments
    db_path, method = get_args()

    # Load embeddings from the Chroma database
    embeddings, idx = load_embeddings(db_path)

    # Reduce dimensionality of embeddings to 2D
    coordinates = reduce_dimensionality(embeddings, method)

    # Save 2D coordinates to a .npy file
    save_2d_coordinates(coordinates, idx, db_path, method)

