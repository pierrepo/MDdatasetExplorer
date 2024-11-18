
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

### Create Datasets

To create the datasets, you need to run the following command:

```bash
python src/create_datasets.py --dataset_infos <path_to_dataset_infos> --file_infos <path_to_file_infos> --gro_infos <path_to_gro_infos> --mdp_infos <path_to_mdp_infos>
```

Arguments :
- `dataset_infos` : Path to the parquet file containing the dataset information (id, origin, title, keywords, description, etc)
- `file_infos` : Path to the parquet file containing the file information (dataset_id, name, extension, etc)
- `gro_infos` : Path to the parquet file containing the gro information (dataset_id, atome number, presence of molecules, etc)
- `mdp_infos` : Path to the parquetfile containing the mdp information (dataset_id, dt, nsteps, temperature, etc)

This command will generate three distinct datasets, each with increasing levels of information about the molecular dynamics datasets:

1. **Basic Dataset**: Contains only the title and abstract for each molecular dynamics dataset.

2. **Extended Dataset**: Adds information about the file types and extensions present in each molecular dataset (e.g., .pdb, .gro, .xtc).

3. **Detailed Dataset**: Includes specific parameter values from mdp files (such as simulation time, temperature, or integration steps) alongside the title, abstract, and file extensions. 


### Tf-idf Vector Creation

To create the tf-idf vectors for each dataset created, you need to run the following command:

```bash
python src/create_tfidf.py
```

This command will processes JSON files in the results/datasets directory, computes TF-IDF vectors (Term Frequency-Inverse Document Frequency) for the text data, and stores the resulting vectors in the results/tfidf_vectors directory. Each dataset is processed individually, and its vectors are stored in a dedicated Chroma database under results/tfidf_vectors/chroma_db_<dataset_name>.