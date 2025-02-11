"""Run the MdDatasetExplorer pipeline.

This script runs the pipeline to process datasets, generate embeddings using a specified model, 
create an interactive 2D plot, generate a graph, and launch a Streamlit app to explore datasets.

The pipeline consists of the following steps:
    1. **Create Datasets** (`src/utils/create_datasets.py`): Load and preprocess datasets from Parquet files.
    2. **Generate Embeddings** (`src/utils/create_embeddings.py`): Generate embeddings using the specified model.
    3. **Create Interactive Plot** (`src/utils/create_interactive_plot.py`): Reduce the dimensionality of the embeddings, cluster them, and create an interactive 2D plot.
    5. **Launch Streamlit App** (`src/utils/streamlit_app.py`): Run a Streamlit app to explore datasets through two approaches:
       - Visualization of dimensionality-reduced embeddings (2D interactive plot).
       - Graph-based exploration where connections between data points represent their distances in the embedding space.


Usage:
======
    python src/run_pipeline.py --model-name <model_name> --dataset-name <dataset_name> 
                                --reduction-method <reduction-method> --cluster-method <cluster_method>

Arguments:
==========
    - `model_name` (str): Name of the model to use for embeddings generation. Choose one of the following:
      ["BERT", "SciBERT", "BioBERT", "SciNCL", "SBERT", "PubMedBERT", "all-MiniLM-L6-v2"]
    - `dataset_name` (str): Name of the dataset to use for embeddings generation. Choose one of the following:
      ["basic", "extended", "detailed"]
    - `reduction_method` (str): Dimensionality reduction technique to apply. Choose one of the following:
      ["umap", "tsne"]
    - `cluster_method` (str): Clustering algorithm to apply. Choose one of the following:
      ["knn", "hdbscan"]

Example:
========
    python src/run_pipeline.py --model_name "all-MiniLM-L6-v2" --dataset_name "basic" --reduction_method "umap" --cluster_method "hdbscan"

This command will : 
    1. Create datasets as JSON files in the `results/datasets` directory.
    2. Create embeddings using the `all-MiniLM-L6-v2` model for the `extended` dataset in the `results/embeddings` directory.
    3. Create an interactive plot (HTML) of the embeddings reduced using UMAP and clustered with hdbscan in the `results/2d_projections` directory.
    5. Launch a Streamlit app to explore the datasets.

"""


# METADATAS
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import subprocess
from typing import List
import argparse

from loguru import logger


# CONSTANTS
MODEL_EMBEDDINGS_NAMES = [
    "BERT", "SciBERT", "BioBERT", "SciNCL", "SBERT", 
    "PubMedBERT", "all-MiniLM-L6-v2"
]
DATASET_NAMES = ["basic", "extended", "detailed"]
REDUCTION_METHODS = ["umap", "tsne"]
CLUSTER_METHODS = ["knn", "hdbscan"]


# FUNCTIONS
def run_command(command: List[str]) -> None:
    """
    Run a shell command and log its progress.

    Parameter:
    ----------
        command (List[str]): Command to run as a list of arguments.
            
    Raises:
    -------
        subprocess.CalledProcessError: If the command fails.
    """
    command_str = " ".join(command)
    logger.debug(f"Running command: {command_str}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command_str}")
        logger.error(f"Error: {e}")
        raise


def main(model_name: str, dataset_name: str, reduction_method: str, cluster_method: str) -> None:
    """
    Main function to execute the pipeline.

    Parameter:
    ----------
    model_name : str
        Name of the model to use for embeddings generation.
    dataset_name : str
        Name of the dataset to use for embeddings generation.
    reduc_method : str
        Dimensionality reduction technique to apply.
    cluster_method : str
        Clustering algorithm to apply.   
    """
    logger.info(" === Starting MdDatasetExplorer pipeline === ")
    # Steps to execute
    try:
        # Step 1: Create datasets
        logger.info("Step 1: Creating datasets...")
        run_command([
            "python", "src/utils/create_datasets.py",
            "--dataset_infos", "data/datasets.parquet",
            "--file_infos", "data/files.parquet",
            "--gro_infos", "data/gromacs_gro_files.parquet",
            "--mdp_infos", "data/gromacs_mdp_files.parquet",
            "--engine_info", "data/file_types.yml"
        ])

        # Step 2: Create embeddings using the specified model
        logger.info(f"Step 2: Creating embeddings using model {model_name}...")
        run_command([
            "python", "src/utils/create_embeddings.py",
            "--model_name", model_name,
            "--dataset_file_name", f"{dataset_name}_dataset.json"
        ])

        # Step 3: Create interactive plot based on embeddings
        logger.info("Step 3: Creating interactive plot based on embeddings...")
        run_command([
            "python", "src/utils/create_interactive_plot.py",
            "--db-path", f"results/embeddings/chroma_db_{dataset_name}_dataset_{model_name}",
            "--reduction-method", reduction_method,
            "--cluster-method", cluster_method
        ])

        # Step 4: Run Streamlit app
        logger.info("Step 4: Running Streamlit app...")
        run_command([
            "streamlit", "run", "src/utils/streamlit_app.py", "--",
            "--db-path",
            f"results/embeddings/chroma_db_{dataset_name}_dataset_{model_name}",
            "--html-path",
            f"results/2d_projections/plot_chroma_db_{dataset_name}_dataset_{model_name}_{reduction_method}_{cluster_method}.html"
        ])

        # End of pipeline
        logger.success(" === MdDatasetExplorer pipeline completed successfully ===")
    except Exception as e:
        logger.error(f"Pipeline failed. Error: {e}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the MdDatasetExplorer pipeline.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for embeddings generation.", choices=MODEL_EMBEDDINGS_NAMES)
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use for embeddings generation.", choices=DATASET_NAMES)
    parser.add_argument("--reduction_method", type=str, required=True, help="Dimensionality reduction technique to apply.", choices=REDUCTION_METHODS)
    parser.add_argument("--cluster_method", type=str, required=True, help="Clustering algorithm to apply.", choices=CLUSTER_METHODS)
    args = parser.parse_args()

    # Run the pipeline
    main(args.model_name, args.dataset_name, args.reduction_method, args.cluster_method)