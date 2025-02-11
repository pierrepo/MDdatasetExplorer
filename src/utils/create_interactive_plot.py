"""Script to Generate Interactive 2D Projections of Embeddings with Clustering.

This script processes a Chroma database of high-dimensional embeddings, reduces their dimensionality 
to two dimensions using UMAP or t-SNE, performs clustering using either K-Nearest Neighbors (KNN) 
or HDBSCAN, and generates an interactive 2D plot using `datamapplot`. The plot includes clusters 
annotated with descriptive names generated by a language model (LLM, OpenAI). The final plot is 
saved as an HTML file in the `results/2d_projections` directory.

Usage
=====
    python src/utils/create_interactive_plot.py --db-path <path_to_db> --reduction-method <reduction_method> --cluster-method <cluster_method>

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
    python src/utils/create_interactive_plot.py --db-path results/embeddings/chroma_db_extended_dataset_all-MiniLM-L6-v2 --reduction-method umap --cluster-method hdbscan

This command will:
- Load the embeddings from the Chroma database located at `results/embeddings/chroma_db_extended_dataset_all-MiniLM-L6-v2`.
- Reduce the dimensionality of the embeddings to 2D using UMAP.
- Perform clustering using HDBSCAN.
- Generate descriptive cluster names using GPT-4 (OpenAI) based on the titles of the embeddings in each cluster.
- Save coordinates of the reduced embeddings and cluster labels to `plot_extended_dataset_all-MiniLM-L6-v2_umap_hdbscan.npy`
- Generate an interactive 2D plot using `datamapplot` with clustered data.
- Save the HTML plot to `results/2d_projections/plot_chroma_db_extended_dataset_all-MiniLM-L6-v2_umap_hdbscan.html`.
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
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Dict, Union
from dotenv import load_dotenv

import umap
import chromadb
import tiktoken
import datamapplot as dmp
from loguru import logger
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


# CONSTANTS
OUT_DIR = "results/2d_projections"
REDUC_METHODS = ["umap", "tsne"]
N_COMPONENTS = 2
CLUSTER_METHODS = ["knn", "hdbscan"]
SAVE_NPY = True


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
    parser.add_argument("--reduction-method", type=str, required=True, choices=REDUC_METHODS, help="Dimensionality reduction method to use.")
    parser.add_argument("--cluster-method", type=str, required=True, choices=CLUSTER_METHODS, help="Clustering method to use.")
    args = parser.parse_args()

    logger.debug(f"db-path: {args.db_path}")
    logger.debug(f"reduction-method: {args.reduction_method}")
    logger.debug(f"cluster-method: {args.cluster_method}")
    logger.success("Command-line arguments parsed successfully. \n")
    return args.db_path, args.reduction_method, args.cluster_method


def load_embeddings(db_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        - metadatas : Array of metadata.
    """
    logger.info("Loading embeddings from the Chroma database...")
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    if not collections:
        raise ValueError(f"No collections found in the database at {db_path}.")

    collection_name = client.list_collections()[0]
    db = client.get_collection(name=collection_name)

    data = db.get(include=["embeddings", "metadatas"])
    embeddings = data.get("embeddings", None)
    labels = data.get("ids", None)
    labels_idx = [label.split("_")[0] for label in labels]
    metadatas = data.get("metadatas", None)
    if embeddings is None:
        raise ValueError(f"No embeddings found in the collection '{collection_name}'.")

    logger.success(f"{len(embeddings)} embeddings loaded successfully. \n")
    return np.array(embeddings), np.array(labels_idx), np.array(metadatas)


def reduce_dimensionality(embeddings: np.array, method: str) -> np.ndarray:
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
    # Check if the cluster ID is -1 (representing outliers)
    if cluster_id == -1:
        return "Others"
    
    # Filter out titles that belong to the current cluster
    cluster_titles = [titles[i] for i in range(len(titles)) if clusters[i] == cluster_id and clusters[i] != -1]

    # Create a string of the filtered titles
    titles_str = ", ".join(cluster_titles)

    # Prepare the prompt for the LLM
    prompt_template = """
    You are an expert in the field of molecular dynamics (MD) and have been asked to name a new cluster of MD datasets.
    Generate a brief (2 or 3 words max) and descriptive name for the following cluster based on the titles: {titles}.
    Ensure that Dynamics, Cluster, Realm, Simulation words are not strictly included in the name.
    Ensure the generated name is not identical to any of the following previously generated names: {used_titles}.
    """

    prompt = PromptTemplate(input_variables=["titles", "used_titles"], template=prompt_template)
    # Verify tokens limitation (16385 tokens)
    enc = tiktoken.encoding_for_model("gpt-4o")
    prompt_tokens = enc.encode(str(prompt))
    if len(prompt_tokens) > 16385:
        to_remove = (len(prompt_tokens) - 16385) *  4
        prompt = prompt_template[:-to_remove]

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


def generate_cluster_names(
    clusters: np.ndarray, 
    metadatas: List[Dict[str, Union[str, int]]]
) -> np.ndarray:
    """
    Generate descriptive names for each cluster, then map each coordinate to its respective cluster name.

    Parameters
    ----------
    clusters : np.ndarray
        Array of cluster labels.
    metadatas : list
        List of metadata dictionaries for each embedding.

    Returns
    -------
    np.ndarray
        Array of descriptive names for each unique cluster.
    """
    logger.info("Creating cluster names for each cluster...")
    used_titles = set()
    cluster_names = []
    titles = [metadata.get("title", "Unknown Title") for metadata in metadatas]
    
    for cluster_id in tqdm(np.unique(clusters), desc="Generating Cluster Names", unit="cluster"):
        cluster_name = get_cluster_name(cluster_id, titles, clusters, used_titles)
        cluster_names.append(cluster_name)
    logger.debug(f"Cluster names: {cluster_names}")

    # Map each coordinate to its respective cluster name
    cluster_name_map = np.array([cluster_names[cluster] for cluster in clusters])
    logger.success("Cluster names generated successfully. \n")

    return cluster_name_map


def save_cluster_data_to_npy(
    idx: np.ndarray, 
    coords: np.ndarray, 
    clusters: np.ndarray, 
    cluster_names: np.ndarray, 
    metadatas: np.ndarray, 
    db_path: str) -> None:
    """
    Save the 2D coordinates, cluster IDs, cluster names, metadata, and indices to a .npy file.

    Parameters
    ----------
    idx : np.ndarray
        Array of indices (embedding IDs).
    coords : np.ndarray
        Array of 2D coordinates of the embeddings.
    clusters : np.ndarray
        Array of cluster labels.
    cluster_names : np.ndarray
        Array of cluster names corresponding to each embedding.
    metadatas : np.ndarray
        Array of metadata dictionaries for each embedding.
    db_path : str
        Path to the Chroma database.
    """
    # Prepare the output directory
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Prepare the output file name
    file_name = f"{os.path.basename(db_path)}_{reduction_method}_{cluster_method}.npy"
    out_file = os.path.join(OUT_DIR, file_name)

    # Create an array that combines idx, coordinates, cluster ID, cluster name, and metadata
    all_data = np.column_stack((idx, coords, clusters, cluster_names, metadatas))

    # Save the combined data to a .npy file
    np.save(out_file, all_data)
    
    logger.success(f"Coords, clusters and metadatas saved to {out_file} successfully. \n")


def plot_and_save_2d_clusters(
    coords: np.ndarray, 
    metadatas: np.ndarray, 
    cluster_names: np.ndarray, 
    db_path : str
) -> None:
    """
    Generate an interactive 2D scatter plot of clusters and save it as an HTML file.

    Parameters
    ----------
    coords : np.ndarray
        2D coordinates of the embeddings.
    metadatas : list
        List of metadata dictionaries for each embedding.
    cluster_names : np.ndarray
        Array of descriptive names for each cluster.
    db_path : str
        Path to the Chroma database.
    """
    logger.info("Creating interactive plot...")

    # Prepare data for datamapplot
    titles = np.array([meta.get('title', 'No Title') for meta in metadatas])
    urls = np.array([str(meta.get('url', 'No URL')) for meta in metadatas])
    descriptions = np.array([meta.get('description', 'No Description') for meta in metadatas], dtype=object)
    authors = np.array([meta.get('author', 'No Authors') for meta in metadatas], dtype=object)
    keywords = np.array([meta.get('keywords', 'No Keywords') for meta in metadatas], dtype=object)
    origins = np.array([meta.get('origin', 'No Origin') for meta in metadatas], dtype=object)
    date_creations = np.array([meta.get('date_creation', 'No Date').split("-")[0] for meta in metadatas], dtype=object)
    software_engines = np.array([meta.get('software_engines', 'No Software Engine')  for meta in metadatas], dtype=object)
    first_software_engines = np.array([eval(engine)[0] if engine != 'No Software Engine' else engine for engine in software_engines], dtype=object)

    # create dataframe for extra data
    extra_data = pd.DataFrame({
        'idx': idx,
        'cluster_name': cluster_names,
        'title': titles,
        'url': urls,
        'description': descriptions,
        'authors': authors,
        'keywords': keywords,
        'origin': origins,
        'date_creation': date_creations,
        'software_engine': first_software_engines
    })

    badge_css = """
        border-radius:6px;
        width:fit-content;
        max-width:75%;
        margin:2px;
        padding: 2px 10px 2px 10px;
        font-size: 10pt;
    """
    hover_text_template = f"""
    <div>
        <div style="background-color:grey;color:#fff;{badge_css}">{{cluster_name}}</div>
        <div style="font-size:11pt;padding:2px;"><B>Title</B>: {{hover_text}}</div>
        <div style="font-size:9pt;padding:2px;"> <B>Author(s)</B>: {{authors}}</div>
        <div style="font-size:9pt;padding:2px;""> <B>Keywords</B>: {{keywords}}</div>
        <div style="font-size:10pt;padding:2px;"> <B>Description</B>: {{description}}</div>
    </div>
    """

    # Create the plot
    plot = dmp.create_interactive_plot(
            coords,
            cluster_names,
            hover_text = titles,
            initial_zoom_fraction=0.9,
            cluster_boundary_line_width=6,
            extra_point_data= extra_data,
            enable_search=True,
            search_field="title",
            hover_text_html_template=hover_text_template,
            cluster_boundary_polygons=True,
            colormaps={"Date": extra_data.date_creation, "Software": extra_data.software_engine, "Origin": extra_data.origin},
            on_click="window.open(`{url}`)",
    )
    
    # Save the plot as an HTML file
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    base_name = os.path.basename(db_path)
    full_output_path = os.path.join(OUT_DIR, f"plot_{os.path.splitext(base_name)[0]}_{reduction_method}_{cluster_method}.html")
    plot.save(full_output_path)
    logger.success(f"Interactive plot saved to {full_output_path} successfully. \n")


# MAIN PROGRAM
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Parse command-line arguments
    db_path, reduction_method, cluster_method = get_args()

    # Load embeddings from the Chroma database
    embeddings, idx, metadatas = load_embeddings(db_path)

    # Reduce dimensionality of embeddings to 2D
    coords = reduce_dimensionality(embeddings, reduction_method)

    # Perform clustering
    if cluster_method == "knn":
        clusters = perform_knn_clustering(coords)
    elif cluster_method == "hdbscan":
        clusters = perform_hdbscan_clustering(coords)
    
    # Get cluster names
    cluster_names = generate_cluster_names(clusters, metadatas)

    # Save cluster data to .npy file
    if SAVE_NPY:
        save_cluster_data_to_npy(idx, coords, clusters, cluster_names, metadatas, db_path)
    
    # Generate interactive visualization and save it as an HTML file
    plot_and_save_2d_clusters(coords, metadatas, cluster_names, db_path)

