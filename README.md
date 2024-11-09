
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

### Add Groq API key

Create an `.env` file with a [valid Groq key](https://console.groq.com/keys):
```text
OPENAI_API_KEY=<your-groq-api-key>
```

> Remark: This `.env` file is ignored by git.


### Create Datasets

To create the datasets, you need to run the following command:

```bash
python src/create_datasets.py
```

This command will generate three distinct datasets, each with increasing levels of information in the  [`data`](link) folder : 

1. **Basic Dataset**: Contains only the title and abstract for each molecular dynamics dataset.

2. **Extended Dataset**: Adds information about the file types and extensions present in each molecular dataset (e.g., .pdb, .gro, .xtc).

3. **Detailed Dataset**: Includes specific parameter values from mdp files (such as simulation time, temperature, or integration steps) alongside the title, abstract, and file extensions. 

### Embedding Creation

To create the embeddings, you need to run the following command:

```bash
python src/create_embeddings.py --dataset <dataset_file_name> --model <embedding_model_name>
```

This command will generate a set of embeddings stored in the [`embeddings`](link) folder.


### Graph Creation

To construct a similarity graph based on dataset embeddings, execute:

```bash
python src/create_graph.py --embeddings <embeddings_file_name> --threshold <threshold_value>
```

This command will generate a graph stored in the [`graphs`](link) folder.


## Usage (web interface)

To explore the dataset graph interactively, use the Streamlit app:

```bash
streamlit run src/streamlit_app.py
```

This will run the Streamlit app in your web browser. 
