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
conda env create -f environment.yml
```

### Activate the Conda environment

```bash
conda activate mddatasetexplorer-env
```

### Export OpenAI API key

```bash
export OPENAI_API_KEY=XXXXX...
```

## Usage

To load MDdatasetExplorer, run the following command:

```bash
python src/run_pipeline.py --model_name <model_name> --dataset_name <dataset_name> --reduction_method <reduction_method> --cluster_method <cluster_method>
```

Where:

- `model_name` is the name of the model to use for embeddings creation.
- `dataset_name` is the name of the dataset to explore. Choose between `basic`, `extended` and `detailed`
- `reduction_method` is the method to use for dimensionality reduction. Choose between `umap` and `tsne`.
- `cluster_method` is the method to use for clustering. Choose between `knn` and `dbscan`.

Example:

```bash
python src/run_pipeline.py --model_name "all-MiniLM-L6-v2" --dataset_name "basic" --reduction_method "umap" --cluster_method "hdbscan"
```

This command will:

1. **Create datasets** (run `src/utils/create_datasets.py`): Load and preprocess datasets from Parquet files as JSON files in the `results/datasets` directory.
2. **Create embeddings** (run `src/utils/create_embeddings.py`): Create embeddings using the `all-MiniLM-L6-v2` model for the `basic` dataset.
3. **Create an interactive plot** (run `src/utils/create_interactive_plot.py`): Create the html plot of the embeddings reduced using UMAP and clustered with hdbscan in the `results/2d_projections` directory.
4. **Create Streamlit app** (run `src/utils/streamlit_app.py`): Launch a Streamlit app to explore the datasets.
