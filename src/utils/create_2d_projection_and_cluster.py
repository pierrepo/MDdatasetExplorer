"""
Script to Generate 2D Projections of Embeddings, Cluster Them, and Generate Cluster Names using LLM.

This script takes a Chroma database of high-dimensional embeddings, reduces their dimensionality 
to two dimensions using UMAP or t-SNE, performs clustering using either K-Nearest Neighbors (KNN) 
or HDBSCAN, and then generates a descriptive name for each cluster using a language model (LLM, OpenAI).
The results are saved in a `.npy` file with columns for the embedding ID, 2D coordinates, cluster ID, 
and cluster name.

Usage
=====
    python src/utils/create_2d_projection_and_cluster.py --db-path <path_to_db> --reduction-method <reduction_method> --cluster-method <cluster_method>

Arguments
=========
    --db-path: str
        Path to the Chroma database containing the embeddings to visualize and cluster.
    --reduction-method: str
        Dimensionality reduction method to use. Options are 'umap' or 'tsne'.
    --cluster-method: str
        Clustering method to use. Options are 'knn' or 'hdbscan'.

Example
=======
    python src/utils/create_2d_projection_and_cluster.py --db-path results/embeddings/chroma_db_extended_dataset_all-MiniLM-L6-v2 --reduction-method umap --cluster-method hdbscan

This command will:
- Load the embeddings from Chroma "results/embeddings/chroma_db_extended_dataset_all-MiniLM-L6-v2" 
- Reduce their dimensionality to 2D using UMAP.
- Perform clustering using HDBSCAN.
- Generate a name for each cluster based on the titles of the embeddings in that cluster using GPT-4o (OpenAI).
- Save the results (embedding ID, 2D coordinates, cluster ID, and cluster name) to "results/2d_projections/chroma_db_extended_all-MiniLM-L6-v2_umap_hdbscan.npy".
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
from typing import Tuple
from dotenv import load_dotenv

import umap
import chromadb
from loguru import logger
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


# CONSTANTS
OUT_DIR = "results/2d_projections"
METHODS = ["umap", "tsne"]
N_COMPONENTS = 2
CLUSTER_METHODS = ["knn", "hdbscan"]



# FUNCTIONS
def get_args() -> Tuple[str, str, str]:
    """Parse command-line arguments.

    Returns
    -------
    Tuple[str, str, str]
        - db_path: Path to the Chroma database containing the embeddings to visualize.
        - method: The dimensionality reduction method (either 'umap' or 'tsne').
        - cluster_method: The clustering method (either 'knn' or 'hdbscan').
    """
    logger.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="Generate 2D projections of embeddings for visualization.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to the Chroma database containing the embeddings to visualize.")
    parser.add_argument("--reduction-method", type=str, required=True, choices=METHODS, help="Dimensionality reduction method to use.")
    parser.add_argument("--cluster-method", type=str, required=True, choices=CLUSTER_METHODS, help="Clustering method to use.")
    args = parser.parse_args()

    logger.debug(f"db-path: {args.db_path}")
    logger.debug(f"reduction-method: {args.reduction_method}")
    logger.debug(f"cluster-method: {args.cluster_method}")
    logger.success("Command-line arguments parsed successfully. \n")
    return args.db_path, args.reduction_method, args.cluster_method


def load_embeddings(db_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load embeddings and IDs from a Chroma database.

    Parameters
    ----------
    db_path : str
        Path to the Chroma database.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, list]
        - embeddings: Array of embeddings.
        - labels: Array of labels/IDs.
        - titles : list of titles.
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
    metadatas = data.get("metadatas", None)
    titles = [metadata.get("title", "Unknown Title") for metadata in metadatas]
    if embeddings is None:
        raise ValueError(f"No embeddings found in the collection '{collection_name}'.")

    logger.success(f"{len(embeddings)} embeddings loaded successfully. \n")
    return np.array(embeddings), np.array(labels_idx), titles


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
    
    coords = reducer.fit_transform(embeddings)
    logger.success(f"{len(coords)} embeddings reduced to 2D successfully. \n")

    return coords


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


def perform_knn_clustering(coordinates: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """Perform KNN clustering on the given 2D coordinates.
    
    This function performs clustering by finding the nearest neighbors for each point in the 2D space 
    and grouping points based on proximity.

    Parameters
    ----------
    coordinates : np.ndarray
        2D coordinates of the embeddings.
    n_neighbors : int, optional
        The number of neighbors to consider when clustering (default is 10).

    Returns
    -------
    np.ndarray
        Array of cluster labels for each embedding based on the KNN clustering.
    """
    logger.info("Performing KNN clustering...")
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(coordinates)
    _distances, indices = knn.kneighbors(coordinates)
    clusters = indices[:, 1]
    logger.success(f"Found {len(np.unique(clusters))} clusters successfully. \n")
    return clusters


def perform_hdbscan_clustering(coordinates: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """Perform HDBSCAN clustering on the given 2D coordinates.
    
    This function performs density-based clustering using HDBSCAN to identify clusters of varying densities.

    Parameters
    ----------
    coordinates : np.ndarray
        2D coordinates of the embeddings.
    min_cluster_size : int, optional
        Minimum size of a cluster (default is 5).

    Returns
    -------
    np.ndarray
        Array of cluster labels for each embedding based on the HDBSCAN clustering.
    """
    logger.info("Performing HDBSCAN clustering...")
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = hdbscan.fit_predict(coordinates)
    logger.success(f"Found {len(np.unique(clusters))} clusters successfully. \n")
    return clusters


def get_cluster_name(cluster_id: int, titles: list, clusters: list, used_titles: set) -> str:
    """
    Generate a descriptive name for the cluster using an LLM (OpenAI via LangChain).
    
    This function takes a cluster ID, extracts the titles of the embeddings in that cluster, 
    and uses a language model to generate a brief, descriptive name for the cluster based on those titles,
    while avoiding redundancy with titles already used by other clusters.

    Parameters
    ----------
    cluster_id : int
        The cluster ID for which to generate the name.
    titles : list
        List of titles of all embeddings.
    clusters : list
        List of cluster assignments for all embeddings.
    used_titles : set
        Set of titles that have already been used for other clusters.

    Returns
    -------
    str
        Descriptive name for the cluster.
    """
    # Filter out titles that belong to the current cluster
    cluster_titles = [titles[i] for i in range(len(titles)) if clusters[i] == cluster_id]

    # Create a string of the filtered titles
    titles_str = ", ".join(cluster_titles)

    # Prepare the prompt for the LLM
    prompt_template = """
    Generate a brief (2 or 3 words max) and descriptive name for the following cluster based on the titles: {titles}.
    Ensure the generated name is not identical to any of the following previously generated names: {used_titles}.
    """

    prompt = PromptTemplate(input_variables=["titles", "used_titles"], template=prompt_template)
    
    # Set up the LLM and chain
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    chain = prompt | llm | StrOutputParser()

    # Generate the cluster name
    cluster_name = chain.invoke({"titles": titles_str, "used_titles": ", ".join(used_titles)})

    # Clean up and return the cluster name
    cluster_name = cluster_name.strip()
    
    # Add the new cluster name to the used_titles set to avoid reusing it in the future
    used_titles.add(cluster_name)

    return cluster_name


def save_2d_coords_and_clusters(coords: np.ndarray, idx: np.array, clusters: np.ndarray, titles: list, db_path: str, reduction_method: str, cluster_method: str) -> None:
    """Save the 2D coordinates and cluster labels to a .npy file, including the descriptive cluster names.

    Parameters
    ----------
    coords : np.ndarray
        2D coordinates of the embeddings.
    idx : np.array
        Array of IDs.
    clusters : np.ndarray
        Array of cluster labels.
    titles : list
        List of titles of all embeddings.
    db_path : str
        Path to the Chroma database.
    reduction_method : str
        Dimensionality reduction method used ('umap' or 'tsne').
    cluster_method : str
        Clustering method used ('knn' or 'hdbscan').
    """
    logger.info("Saving 2D coordinates and cluster labels to a .npy file...")
    
    # Prepare the output file name
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    out_file = os.path.join(OUT_DIR, f"{os.path.basename(db_path)}_{reduction_method}_{cluster_method}.npy")
    
    # Initialize used_titles set
    used_titles = set()

    # Generate cluster names for each cluster ID while avoiding redundancy
    cluster_names = []
    for cluster_id in np.unique(clusters):
        cluster_name = get_cluster_name(cluster_id, titles, clusters, used_titles)
        cluster_names.append(cluster_name)
    
    # Create a new array with columns: ID, Coordinates, Cluster, Cluster Name
    all_data = np.hstack((idx.reshape(-1, 1), coords, clusters[:, np.newaxis]))
    # Map cluster names to the corresponding cluster labels
    cluster_names_array = np.array([cluster_names[cluster] for cluster in clusters])[:, np.newaxis]
    # Combine the data and cluster names
    all_data_with_names = np.hstack((all_data, cluster_names_array))
    
    # Save the result to a new .npy file
    np.save(out_file, all_data_with_names)
    logger.success(f"2D coordinates and cluster labels saved to {out_file} successfully. \n")


# MAIN PROGRAM
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Parse command-line arguments
    db_path, reduction_method, cluster_method = get_args()

    # Load embeddings from the Chroma database
    embeddings, idx, titles = load_embeddings(db_path)

    # Reduce dimensionality of embeddings to 2D
    coords = reduce_dimensionality(embeddings, reduction_method)

    # Perform clustering
    if cluster_method == "knn":
        clusters = perform_knn_clustering(coords)
    elif cluster_method == "hdbscan":
        clusters = perform_hdbscan_clustering(coords)
    
    # Save 2D coordinates to a .npy file
    save_2d_coords_and_clusters(coords, idx, clusters, titles, db_path, reduction_method, cluster_method)

