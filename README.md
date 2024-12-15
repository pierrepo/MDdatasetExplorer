
<h1 align="center">
  <img style="vertical-align:middle; width:70%; position:fixed;">
  MDdatasetExplorer
</h1>

<p align="center" style="width: 500px;">
  <i> A Tool for Exploring and Clustering Open Molecular Dynamics Datasets through Metadata Similarity and Graph Modeling.
  </i>
</p>

<p align="center">
    <img alt="Made with Python" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=%23539fc9">
    <img alt="BSD-3 Clause License" src="https://img.shields.io/github/license/pierrepo/MDdatasetExplorer?style=flat&color=%23539fc9&link=https%3A%2F%2Fgithub.com%2Fpierrepo%2FMDdatasetExplorer%2Fblob%2Fmain%2FLICENSE">
</p>


## Setup

To install MDdatasetExplorer and its dependencies, you need to perform the following steps:

### Clone the repository

```bash
git clone https://github.com/pierrepo/MDdatasetExplorer.git
cd MDdatasetExplorer
```

### Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create a Conda environment

```bash
conda config --add channels conda-forge
conda config --add channels defaults
conda env create -f environment.yml
```

### Activate the Conda environment

```bash
conda activate mddatasetexplorer-env
```

## Usage

To run the MDdatasetExplorer, you need to run the following command:

```bash
  python src/run_pipeline.py --model_name <model_name> --dataset_name <dataset_name>
```

Where : 
- `model_name` is the name of the model to use for embeddings creation.
- `dataset_name` is the name of the dataset to explore.

Example :

```bash
  python src/run_pipeline.py --model_name "all-MiniLM-L6-v2" --dataset_name "extended"
```

This command will : 
  1. **Create datasets** (`src/utils/create_datasets.py`): Load and preprocess datasets from Parquet files.
  2. **Create TF-IDF vectors** (`src/utils/create_tfidf_vectors.py`): Generate TF-IDF vectors from the preprocessed datasets. [TODO]
  3. **Create embeddings** (`src/utils/create_embeddings.py`): Generate embeddings using the specified model.
  4. **Create graph** (`src/utils/create_graph.py`): Create a graph based on the embeddings. [TODO]
  5. **Create Streamlit app** (`src/utils/streamlit_app.py`): Run Streamlit app to explore the datasets.
