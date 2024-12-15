""" Streamlit application for MDdatasetExplorer.

Usage
=====
    streamlit run src/utils/streamlit_app.py --db-path <path_to_chroma_db>

Argument:
=========
    --db-path : str
        Path to the Chroma database.

Example:
========
    streamlit run src/utils/streamlit_app.py -- --db-path results/embeddings/chroma_db_extended_dataset_all-MiniLM-L6-v2
"""


# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import json
import base64
import argparse

import chromadb
import pandas as pd
import numpy as np
from loguru import logger
import streamlit as st


# FUNCTIONS: UTILITIES
def get_arg() -> str:
    """Get the command-line arguments."""
    parser = argparse.ArgumentParser(description="Streamlit application for MDdatasetExplorer.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to the Chroma database.")
    args = parser.parse_args()
    return args.db_path


def load_embeddings(db_path: str):
    """
    Load embeddings and IDs from a Chroma database.

    Parameters
    ----------
    db_path : str
        Path to the Chroma database.

    Returns
    -------
    Tuple[object, np.ndarray, np.ndarray, np.ndarray]
        - db: ChromaDB collection object.
        - embeddings: Array of embeddings.
        - labels: Array of labels/IDs.
        - titles: Array of titles from metadata.
    """
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

    return db, np.array(embeddings), np.array(labels_idx), np.array(titles)


def search_similarity_in_db(db, query, top_k=10):
    """
    Search for similar embeddings in the database using a query.

    Parameters
    ----------
    db : object
        ChromaDB collection object.
    query : str
        The query text.
    top_k : int, optional
        Number of top results to retrieve, by default 10.

    Returns
    -------
    list of dict
        List of top-k results with metadata and similarity scores.
    """
    results = db.query(query_texts=[query], n_results=top_k)
    return results


def process_search_results(results):
    """Process search results to prepare a DataFrame for display."""
    similarities = results["distances"][0]
    documents = results["documents"][0]
    dataset_info = []
    for doc in documents:
        doc_data = json.loads(doc)  # parse individual JSON string
        dataset_info.append({
            "title": doc_data["title"],
            "keywords": doc_data["keywords"],
            "date_creation": doc_data["date_creation"],
            "description": doc_data["description"], 
            "authors": doc_data["author"],
            "software_engine": doc_data["software_engines"],
            "origin": doc_data["origin"],
            "url": doc_data["url"]
        })
    df = pd.DataFrame(dataset_info)
    df["similarity"] = similarities
    df_sorted = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)
    return df_sorted


def callback():
    """Callback function to display dataset details in the sidebar."""
    if st.session_state.row:
        selected_metadata = st.session_state.data.iloc[st.session_state.row].astype(str).to_dict()
        details = f"""
            <style>
                .dataset-details {{
                    font-family: Arial, sans-serif;
                    line-height: 1.2;  /* Réduit l'espacement entre les lignes */
                }}
                .dataset-details p {{
                    font-size: 14px;  /* Taille de police normale pour les titres */
                    margin: 5px 0;    /* Réduit l'espacement entre les paragraphes */
                }}
                .dataset-details p strong {{
                    font-weight: bold;  /* Garde les titres en gras */
                }}
                .dataset-details p span {{
                    font-style: italic;  /* Met les valeurs en italique */
                    color: #888;         /* Rend les valeurs plus claires (gris clair) */
                    font-size: 12px;     /* Réduit la taille de la police pour les valeurs */
                }}
            </style>
            <div class="dataset-details">
                <p><strong>URL:</strong> <span>{selected_metadata.get('url', 'N/A')}</span></p>
                <p><strong>Origin:</strong> <span>{selected_metadata.get('origin', 'N/A')}</span></p>
                <p><strong>Date:</strong> <span>{selected_metadata.get('date_creation', 'N/A')}</span></p>
                <p><strong>Authors:</strong> <span>{selected_metadata.get('authors', 'N/A')}</span></p>
                <p><strong>Title:</strong> <span>{selected_metadata.get('title', 'N/A')}</span></p>
                <p><strong>Keywords:</strong> <span>{selected_metadata.get('keywords', 'N/A')}</span></p>
                <p><strong>Software Engine:</strong> <span>{selected_metadata.get('software_engine', 'N/A')}</span></p>
                <p><strong>Description:</strong> <span>{selected_metadata.get('description', 'N/A')}</span></p>
            </div>
        """
        st.write(details)

# FUNCTIONS : CREATE STREAMLIT COMPONENTS
def create_banner():
    """Create and display the application banner."""
    with open("data/img/banner_app.gif", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    banner_html = f"""
    <div class="banner">
        <img src="data:image/gif;base64,{encoded_image}" alt="Banner" style="width:100%; height: 100%; object-fit: cover;">
    </div>
    <style>
        .banner {{
            text-align: center;
            margin: 0;
            position: absolute;
            top: 0px;
            left: 0;
            right: 0;
        }}
    </style>
    """
    st.components.v1.html(banner_html)


def create_sidebar():
    """Create the sidebar with dataset details and About section."""
    with st.sidebar:
        st.header("About")
        st.info("This application is part of the [MDverse project](https://doi.org/10.7554/eLife.90061.3), aimed at exploring molecular dynamics datasets."
                "You can search for datasets, view details, and explore embedding visualizations.")
        
        st.divider()

        sentiment_mapping = ["one", "two", "three", "four", "five"]
        st.header("Feedback")
        st.markdown("How would you rate this app?")
        selected = st.feedback("stars")
        if selected is not None:
            logger.info(f"User feedback: {sentiment_mapping[selected-1]} stars")
            st.toast(f"Thank you for your feedback! You rated the app with {sentiment_mapping[selected]} stars 🌟")


def create_main_page():
    """Create the main page layout with similarity search and embedding visualization."""
    main_container = st.container()

    with main_container:
        # Similarity Search Section
        prompt = st.chat_input("Search datasets by entering a topic, keyword, or title...")

        # Initialize session state if not already present
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = None

        # Handle prompt input and searching for similar datasets
        if prompt:
            with st.spinner("Searching for similar datasets..."):
                db_path = get_arg()  # Assuming get_arg is defined elsewhere
                db, embeddings, labels, titles = load_embeddings(db_path)  # Assuming load_embeddings is defined elsewhere
                top_n = search_similarity_in_db(db, prompt, top_k=30)  # Assuming search_similarity_in_db is defined elsewhere
                df = process_search_results(top_n)  # Assuming process_search_results is defined elsewhere
                
                # Save the search results in session state
                st.session_state.data = df

        # Display search results if available
        if st.session_state.data is not None:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("Top Similar Datasets:")
                st.dataframe(st.session_state.data[["title", "similarity"]])

            with col2:
                # Select dataset
                dataset_selected = st.selectbox("Select a dataset to see details", st.session_state.data.index, format_func=lambda x: st.session_state.data.loc[x, "title"])
                
                # Store the selected dataset's index in session state
                st.session_state.selected_dataset = dataset_selected

                # Retrieve and display selected dataset details
                if st.session_state.selected_dataset is not None:
                    selected_metadata = st.session_state.data.iloc[st.session_state.selected_dataset].astype(str).to_dict()

                    details = f"""
                        <style>
                            .dataset-details {{
                                font-family: Arial, sans-serif;
                                line-height: 1.2;
                            }}
                            .dataset-details p {{
                                font-size: 14px;
                                margin: 5px 0;
                            }}
                            .dataset-details p strong {{
                                font-weight: bold;
                            }}
                            .dataset-details p span {{
                                font-style: italic;
                                color: #888;
                                font-size: 14px;
                            }}
                        </style>
                        <div class="dataset-details">
                            <p><strong>URL:</strong> <a href="{selected_metadata.get('url', '#')}" target="_blank">{selected_metadata.get('url', 'N/A')}</a></p>
                            <p><strong>Origin:</strong> <span>{selected_metadata.get('origin', 'N/A')}</span></p>
                            <p><strong>Date:</strong> <span>{selected_metadata.get('date_creation', 'N/A')}</span></p>
                            <p><strong>Authors:</strong> <span>{selected_metadata.get('authors', 'N/A')}</span></p>
                            <p><strong>Title:</strong> <span>{selected_metadata.get('title', 'N/A')}</span></p>
                            <p><strong>Keywords:</strong> <span>{selected_metadata.get('keywords', 'N/A')}</span></p>
                            <p><strong>Software Engine:</strong> <span>{selected_metadata.get('software_engine', 'N/A')}</span></p>
                            <p><strong>Description:</strong> <span>{selected_metadata.get('description', 'N/A')}</span></p>
                        </div>
                    """
                    st.markdown(details, unsafe_allow_html=True)


def create_footer():
    """Create the footer with feedback and credits."""
    st.markdown(
        """
        <style>
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            margin-top: 50px;
            font-size: 0.9em;
            color: gray;
            backdrop-filter: blur(10px); /* Blur effect */
            border-top: 1px solid rgba(200, 200, 200, 0.5); /* Subtle top border */
            padding: 10px 0;
        }
        </style>
        <footer>
            This application was developed by Essmay Touami and Pierre Poulain as part of the MDverse project. <br>
            The source code is available on <a href="https://github.com/pierrepo/MDdatasetExplorer" target="_blank">GitHub</a> under the BSD 3-Clause license.
        </footer>
        """,
        unsafe_allow_html=True,
    )


# MAIN PROGRAM OF STREAMLIT APPLICATION
if __name__ == "__main__":
    logger.info("--- Starting the Streamlit application ---")
    st.set_page_config(page_title="MDdatasetExplorer", layout="wide", page_icon="⚛", initial_sidebar_state='collapsed')
    create_banner()
    create_sidebar()
    create_main_page()
    create_footer()