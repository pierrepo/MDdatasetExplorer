"""Dataset Creation Script for MDdatasetExplorer.

This script generates multiple dataset files with varying levels of information from a single input dataset in Parquet format.

The generated datasets are:
1. `basic_dataset.json`: Contains only the title and abstract of each molecular dynamics dataset.
2. `extended_dataset.json`: Adds information about file types and extensions present in each molecular dataset (e.g., .pdb, .gro, .xtc).
3. `detailed_dataset.json`: Includes parameters from `.mdp` files (such as simulation time, temperature, etc.), in addition to title, abstract, and file extensions.

Usage:
======
    python src/create_datasets.py --dataset_infos <path_to_dataset_infos> --file_infos <path_to_file_infos> --gro_infos <path_to_gro_infos> --mdp_infos <path_to_mdp_infos>
Arguments:
==========
    --dataset_info : str
        Path to the Parquet file containing the full dataset.
    --file_infos : str
        Path to the Parquet file containing informations about the dataset files.
    --gro_infos : str
        Path to the Parquet file containing informations about the GRO files.
    --mdp_infos : str
        Path to the Parquet file containing informations about the MDP files.

Example:
========
    python src/create_datasets.py --dataset_infos data/datasets.parquet --file_infos data/files.parquet --gro_infos data/gromacs_gro_files.parquet --mdp_infos data/gromacs_mdp_files.parquet
"""


# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import pandas as pd
import json
import argparse
from typing import Dict, List, Tuple
from copy import deepcopy

from loguru import logger

# CONSTANTS
OUTPUT_DIR = "results/datasets"


# FUNCTIONS
def get_args() -> Tuple[str, str, str, str]:
    """Parse command-line arguments.

    Returns
    -------
    dataset_infos : str

    """
    logger.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="Generate datasets with different levels of information.")
    parser.add_argument("--dataset_infos", type=str, required=True, help="Path to the input Parquet file.")
    parser.add_argument("--file_infos", type=str, required=True, help="Path to the directory containing file information.")
    parser.add_argument("--gro_infos", type=str, required=True, help="Path to the directory containing GRO file information.")
    parser.add_argument("--mdp_infos", type=str, required=True, help="Path to the directory containing MDP file information.")
    args = parser.parse_args()

    logger.info(f"Dataset informations: {args.dataset_infos}")
    logger.info(f"File informations: {args.file_infos}")
    logger.info(f"GRO informations: {args.gro_infos}")
    logger.info(f"MDP informations: {args.mdp_infos}")
    logger.success("Parsed the arguments successfully. \n")

    return args.dataset_infos, args.file_infos, args.gro_infos, args.mdp_infos


def save_json(data: List[Dict], filename: str) -> None:
    """Save a list of dictionaries to a JSON file.

    Parameters
    ----------
    data : List[Dict]
        List of dictionaries representing dataset entries.
    filename : str
        The filename for the output JSON file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(f"{OUTPUT_DIR}/{filename}", "w") as f:
        json.dump(data, f, indent=2)
    
    logger.success(f"Saved JSON file to {filename} successfully. \n")


def generate_datasets(dataset_path: str, files_path: str, gro_path: str, mdp_path: str) -> None:
    """Generate datasets with varying levels of detail and save as JSON files.

    Parameters
    ----------
    dataset_path : str
        Path to the Parquet file containing the full dataset.
    files_path : str
        Path to the directory containing informations about the dataset files.
    gro_path : str
        Path to the directory containing informations about the GRO files.
    mdp_path : str
        Path to the directory containing informations about the MDP files.
    """
    logger.info("Generating datasets with different levels of information...")
    # Load the full dataset
    df = pd.read_parquet(dataset_path)

    # Basic Dataset: Title, Keywords, Date of Creation, and Description
    basic_data = [
        {
            "title": row["title"],
            "keywords": row["keywords"],
            "date_creation": row["date_creation"],
            "description": row["description"]
        }
        for _, row in df.iterrows()
    ]
    save_json(basic_data, "basic_dataset.json")


    # Extended Dataset: Basic Dataset + File Extensions
    # Load the file information and join with the main dataset on 'dataset_id'
    files_df = pd.read_parquet(files_path)
    df = df.merge(files_df.groupby("dataset_id")["file_type"].apply(list).reset_index(), on="dataset_id", how="left")
    
    extended_data = deepcopy(basic_data)
    for i, row in enumerate(df.itertuples()):
        # Replace NaN values with "none" in file_extensions column
        file_extension = getattr(row, "file_type", None)
        
        # Check if file_extension is None or empty and assign ["none"]
        if not file_extension or isinstance(file_extension, float):
            file_extension = ["none"]
        
        # Remove 'none' from the list, then remove duplicates by converting it to a set
        file_extension = [ext for ext in file_extension if ext != "none"]
        file_extension = list(set(file_extension))  # Remove duplicates using set
        
        # Store unique, non-"none" file extensions
        extended_data[i]["file_extensions"] = file_extension if file_extension else ["none"]
        
    save_json(extended_data, "extended_dataset.json")


    # Detailed Dataset: Extended Dataset + MDP Parameters
    # Load the GRO and MDP information and join with the main dataset on 'dataset_id'
    gro_df = pd.read_parquet(gro_path).groupby("dataset_id").agg({
        "atom_number": "first",
        "has_protein": "max",
        "has_nucleic": "max",
        "has_glucid": "max",
        "has_lipid": "max",
        "has_water_ion": "max"
    }).reset_index()
    mdp_df = pd.read_parquet(mdp_path).groupby("dataset_id").agg({
        "dt": "first",
        "nsteps": "first",
        "temperature": "first",
        "thermostat": lambda x: next((i for i in x if i not in ["no", "undefined"]), None),
        "barostat": lambda x: next((i for i in x if i not in ["no", "undefined"]), None)
    }).reset_index()

    df = df.merge(gro_df, on="dataset_id", how="left")
    df = df.merge(mdp_df, on="dataset_id", how="left")

    detailed_data = deepcopy(extended_data)
    # List of column names that need default handling
    columns_to_check = [
        "atom_number", "has_protein", "has_nucleic", "has_glucid", "has_lipid", 
        "has_water_ion", "dt", "nsteps", "temperature", "thermostat", "barostat"
    ]

    # Iterate over each row and update detailed_data
    for i, row in df.iterrows():
        # For each column, check if the value is valid and if not, assign a default value
        for col in columns_to_check:
            value = getattr(row, col, "none")
            
            # For certain columns, we also need special checks (thermostat, barostat)
            if col in ["thermostat", "barostat"]:
                value = value if value not in ["no", "undefined"] else "none"
            
            # If the value is a float and NaN (which is common in pandas), replace with "none"
            if isinstance(value, float) and pd.isna(value):
                value = "none"
            
            # Assign the value to the detailed_data list at index i
            detailed_data[i][col] = value
    
    save_json(detailed_data, "detailed_dataset.json")

    logger.success("Generated datasets successfully. \n")


# MAIN PROGRAM
if __name__ == "__main__":
    # Parse command-line arguments
    dataset_infos, file_infos, gro_infos, mdp_infos = get_args()

    # Generate datasets
    generate_datasets(dataset_infos, file_infos, gro_infos, mdp_infos)

