"""Run the MdDatasetExplorer pipeline.

This script runs the pipeline to process datasets, generate embeddings using a specified model and run Streamlit app to explore the datasets.

The pipeline consists of the following steps:
    1. Create datasets (src/utils/create_datasets.py): Load and preprocess datasets from Parquet files.
    2. Create TF-IDF vectors (src/utils/create_tfidf_vectors.py): Generate TF-IDF vectors from the preprocessed datasets. [TODO]
    3. Create embeddings (src/utils/create_embeddings.py): Generate embeddings using the specified model.
    4. Create graph (src/utils/create_graph.py): Create a graph based on the embeddings. [TODO]
    5. Create Streamlit app (src/utils/streamlit_app.py): Run Streamlit app to explore the datasets.

Usage:
======
    python src/run_pipeline.py --model_name <model_name> --dataset_name <dataset_name>

Arguments:
==========
    - model_name (str): Name of the model to use for embeddings generation. Choose one of the following:
    ["BERT", "SciBERT", "BioBERT", "SciNCL", "SBERT", "PubMedBERT", "all-MiniLM-L6-v2"]
    - dataset_name (str): Name of the dataset to use for embeddings generation. Choose one of the following:
    ["basic", "extended", "detailed"]


Example:
========
    python src/run_pipeline.py --model_name "all-MiniLM-L6-v2" --dataset_name "extended"
"""


# METADATA
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


def main(model_name: str, dataset_name: str) -> None:
    """
    Main function to execute the pipeline.

    Parameter:
    ----------
    model_name : str
        Name of the model to use for embeddings generation.
    dataset_name : str
        Name of the dataset to use for embeddings generation.        
    """
    logger.info(" === Starting MdDatasetExplorer pipeline === ")
    # Steps to execute
    try:
        # Check if model name is valid
        if model_name not in MODEL_EMBEDDINGS_NAMES:
            raise ValueError(f"Invalid model name. Choose one of the following: {MODEL_EMBEDDINGS_NAMES}")

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

        # Step 2: Create TF-IDF vectors
        #logger.info("Step 2: Creating TF-IDF vectors...")
        #run_command([
        #    "python", "src/utils/create_tfidf_vectors.py"
        #])

        # Step 3: Create embeddings using the specified model
        logger.info(f"Step 3: Creating embeddings using model {model_name}...")
        run_command([
            "python", "src/utils/create_embeddings.py",
            "--model_name", model_name,
            "--dataset_file_name", f"{dataset_name}_dataset.json"
        ])

        # Step 4: Create graph based on embeddings
        #logger.info("Step 4: Creating graph based on embeddings...")
        #run_command([
        #    "python", "src/utils/create_graph.py",
        #    "--model_name", model_name
        #])


        # Step 5: Run Streamlit app
        logger.info("Step 5: Running Streamlit app...")
        run_command([
            "streamlit", "run", "src/utils/streamlit_app.py", "--",
            "--db-path",
            f"results/embeddings/chroma_db_{dataset_name}_dataset_{model_name}"
        ])

        # End of pipeline
        logger.success(" === MdDatasetExplorer pipeline completed successfully ===")
    except Exception as e:
        logger.error(f"Pipeline failed. Error: {e}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the MdDatasetExplorer pipeline.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for embeddings generation.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use for embeddings generation.")
    args = parser.parse_args()

    # Run the pipeline
    main(args.model_name, args.dataset_name)