"""
Streamlit application for MDdatasetExplorer.

Usage
=====
    streamlit run src/utils/streamlit_app.py --db-path <path_to_chroma_db> --html-path <path_to_html>

Arguments:
==========
    --db-path : str
        Path to the Chroma database.
    --html-path : str
        Path to the interactive plot HTML file.

Example:
========
    streamlit run src/utils/streamlit_app.py -- --db-path results/embeddings/chroma_db_extended_dataset_all-MiniLM-L6-v2 --html-path results/2d_projections/plot_chroma_db_extended_dataset_all-MiniLM-L6-v2.html
"""


# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import json
import base64
import argparse
import random
from tqdm import tqdm
from typing import Tuple, List, Dict

import chromadb
import pandas as pd
import numpy as np
from loguru import logger
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity


# FUNCTIONS: UTILITIES
def get_args() -> Tuple[str, str]:
    """Get the command-line arguments."""
    logger.info("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description="Streamlit application for MDdatasetExplorer.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to the Chroma database.")
    parser.add_argument("--html-path", type=str, required=True, help="Path to the interactive plot HTML file.")
    args = parser.parse_args()
    logger.debug(f"db-path: {args.db_path}")
    logger.debug(f"html-path: {args.html_path}")
    logger.success("Command-line arguments parsed successfully.")
    return args.db_path, args.html_path


def load_embeddings(db_path: str) -> Tuple[object, np.ndarray, List]:
    """
    Load embeddings and IDs from a Chroma database.

    Parameters
    ----------
    db_path : str
        Path to the Chroma database.

    Returns
    -------
    Tuple[object, np.ndarray, Dict]
        - db: ChromaDB collection object.
        - embeddings: Array of embeddings.
        - metadatas: List of metadata.
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
    _labels_idx = [label.split("_")[0] for label in labels]
    metadatas = data.get("metadatas", None)
    _titles = [metadata.get("title", "Unknown Title") for metadata in metadatas]

    if embeddings is None:
        raise ValueError(f"No embeddings found in the collection '{collection_name}'.")

    return db, np.array(embeddings), metadatas


def search_similarity_in_db(db, query: str, top_k:int=10) -> List[Dict]:
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


def create_graph(embeddings: np.ndarray, metadatas: list, similarity_threshold: float=0.9) -> nx.Graph:
    """
    Create a graph based on cosine similarity between embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        High-dimensional embeddings.
    metadatas : list
        Metadata corresponding to each embedding.
    similarity_threshold : float
        Minimum cosine similarity required to create an edge. Default is 0.9.

    Returns
    -------
    nx.Graph
        Graph object with nodes representing embeddings and edges representing cosine similarity > threshold.
    """
    logger.info("Creating graph based on cosine similarity...")
    graph = nx.Graph()

    # Add nodes with metadata as attributes
    logger.info("Adding nodes to the graph...")
    for i, metadata in enumerate(metadatas):
        graph.add_node(i, **metadata)

    # Compute pairwise cosine similarity
    logger.info("Computing pairwise cosine similarities...")
    similarity_matrix = cosine_similarity(embeddings)

    # Create edges for similarities above the threshold
    logger.info("Adding edges based on similarity threshold...")
    num_nodes = len(embeddings)
    for i in tqdm(range(num_nodes), desc="Processing nodes", unit="node"):
        for j in range(i + 1, num_nodes):  # Avoid duplicate pairs
            if similarity_matrix[i, j] > similarity_threshold:
                graph.add_edge(i, j, weight=similarity_matrix[i, j])

    logger.success(f"Graph created successfully with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges. \n")
    return graph


def visualize_graph_interactive(graph: nx.Graph, similarity_threshold: float, db_path: str):
    """
    Visualize the graph interactively using PyVis.

    Parameters
    ----------
    graph : nx.Graph
        The graph to visualize.
    similarity_threshold : float
        Similarity threshold used for edge creation.
    db_path : str
        Path to the Chroma database (used for naming output file).
    """
    logger.info("Visualizing graph interactively...")
    net = Network(notebook=False)
    net.from_nx(graph)

    # Generate output file path
    base_name = os.path.basename(db_path)
    plot_file = os.path.join("../../results/graphs", f"graph_{os.path.splitext(base_name)[0]}_threshold_{similarity_threshold}.html")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(plot_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save and display the interactive graph
    net.show(plot_file)
    logger.success(f"Interactive graph saved to {plot_file}. Open the file to view the interactive network.")


def community_detection_and_visualization(graph: nx.Graph, similarity_threshold: float=0.9):
    """
    Perform community detection using NetworkX and visualize the result in Streamlit.

    Parameters
    ----------
    graph : nx.Graph
        The graph on which community detection will be performed.
    similarity_threshold : float
        Similarity threshold used for edge creation.
    """
    logger.info("Detecting communities in the graph...")
    
    # Perform community detection using label propagation
    communities = list(nx.community.asyn_lpa_communities(graph))

    # Create a mapping of nodes to their community index
    node_to_community = {node: i for i, community in enumerate(communities) for node in community}

    # Get the 5 largest communities
    community_sizes = [len(community) for community in communities]
    largest_communities = sorted(range(len(community_sizes)), key=lambda i: community_sizes[i], reverse=True)[:5]   
  
    # Generate colors for each community
    num_communities = len(communities)
    color_map = [plt.cm.tab20(i / num_communities) for i in range(num_communities)]
    node_colors = [color_map[node_to_community[node]] for node in graph.nodes()]

    # Visualize the graph with community colors
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(graph, k=0.08, iterations=20)
    nx.draw(
        graph,
        pos,
        ax=ax,
        node_size=40,
        node_color=node_colors,
        font_size=8,
        font_weight="bold",
        edge_color="gray",
        alpha=0.5
    )
    # Add cluster labels
    for i in largest_communities:
        community = communities[i]
        x_coords = [pos[node][0] for node in community]
        y_coords = [pos[node][1] for node in community]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        ax.text(
            centroid_x, centroid_y, 
            f"{i+1}", 
            fontsize=7, fontweight="bold", color="black", 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )

    plt.title(f"Community Detection (Threshold: {similarity_threshold})", fontsize=9)
    # Display the plot in Streamlit
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    # Display the number of communities and their sizes
    download_text = ""

    # Display cluster details
    for i in largest_communities:
        community = communities[i]
        st.write(f"### Cluster {i+1} (Size: {len(community)})")
        
        # Get metadata
        metadata_list = [graph.nodes[node] for node in community]
        
        # Select 10 random metadata titles (if available)
        if metadata_list:
            random_titles = random.sample(metadata_list, min(10, len(metadata_list)))
            titles_text = "\n".join([f"- {meta.get('title', 'No Title')}" for meta in random_titles])
            st.write(f"**Sample Titles:**\n{titles_text}")
        else:
            st.write("No metadata available.")

        # Prepare full metadata for download
        cluster_data = f"Cluster {i+1} (Size: {len(community)})\n"
        cluster_data += "\n".join([f"{meta.get('id', 'No ID')} - {meta.get('title', 'No Title')}" for meta in metadata_list])
        cluster_data += "\n\n"
        download_text += cluster_data

    # Provide a download button
    file_name = "dataset_info.txt"
    st.download_button(
        label="📥 Download Dataset Info",
        data=download_text,
        file_name=file_name,
        mime="text/plain"
    )


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
def create_banner() -> None:
    """Create and display the application banner."""
    with open("data/img/banner_app.gif", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    banner_html = f"""
    <div class="banner">
        <img src="data:image/gif;base64,{encoded_image}" alt="Banner" style="width:100%; height: 40%; object-fit: cover;">
    </div>
    <style>
        .banner {{
            text-align: center;
            margin: 0;
            position: absolute;
            top: 0px;
            left: 0;
            right: 0;
            z-index: -1;
        }}
    </style>
    """
    st.components.v1.html(banner_html)


def create_sidebar() -> None:
    """Create the sidebar with About section and Feedback form."""
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


def display_interactive_plot(html_path: str) -> None:
    """Display the interactive plot.
    
    Parameter:
    ---------
    html_path (str):
        Path to the interactive plot HTML file.
    """
    try:
        with open(html_path, "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    except FileNotFoundError:
        st.error(f"Interactive plot not found at {html_path}. Please ensure the file exists.")


def create_main_page(db_path: str, html_path: str) -> None:
    """Create the main page layout with similarity search and visualization tabs.
    
    Parameter:
    ---------
    db_path (str):
        Path to the Chroma database containing the embeddings.
    html_path (str): 
        Path to the interactive plot HTML file.
    """
    main_container = st.container()

    with main_container:
        # Tabs for the main page
        tab1, tab2, tab3 = st.tabs(["📊 Interactive Plot", "📈 Graph Visualization", "🔍 Similarity Search"])

        with tab1:
            display_interactive_plot(html_path)
        
        with tab3:
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
                    db, embeddings, metadatas = load_embeddings(db_path)
                    top_n = search_similarity_in_db(db, prompt, top_k=50)
                    df = process_search_results(top_n)
                    st.session_state.data = df

            # Display search results
            if st.session_state.data is not None:
                col1, col2 = st.columns([6.5, 5.5])
                with col1:
                    st.write("Top Similar Datasets:")
                    st.dataframe(st.session_state.data[["title", "similarity"]], use_container_width=True)

                with col2:
                    dataset_selected = st.selectbox("Select a dataset to see details", st.session_state.data.index,
                                                     format_func=lambda x: st.session_state.data.loc[x, "title"])
                    st.session_state.selected_dataset = dataset_selected

                    if st.session_state.selected_dataset is not None:
                        selected_metadata = st.session_state.data.iloc[st.session_state.selected_dataset].astype(str).to_dict()
                        details = f"""
                            <div>
                                <p><strong>URL:</strong> <a href="{selected_metadata.get('url', '#')}" target="_blank">{selected_metadata.get('url', 'N/A')}</a></p>
                                <p><strong>Origin:</strong> {selected_metadata.get('origin', 'N/A')}</p>
                                <p><strong>Date:</strong> {selected_metadata.get('date_creation', 'N/A')}</p>
                                <p><strong>Authors:</strong> {selected_metadata.get('authors', 'N/A')}</p>
                                <p><strong>Title:</strong> {selected_metadata.get('title', 'N/A')}</p>
                                <p><strong>Keywords:</strong> {selected_metadata.get('keywords', 'N/A')}</p>
                                <p><strong>Software Engine:</strong> {selected_metadata.get('software_engine', 'N/A')}</p>
                                <p><strong>Description:</strong> {selected_metadata.get('description', 'N/A')}</p>
                            </div>
                        """
                        st.markdown(details, unsafe_allow_html=True)

        with tab2:
            # Create the graph
            _db, embeddings, metadatas = load_embeddings(db_path)
            graph = create_graph(embeddings, metadatas, similarity_threshold=0.98)

            # Visualize the graph interactively
            #visualize_graph_interactive(graph=graph, similarity_threshold=0.9, db_path=db_path)

            # Perform community detection and visualize it
            with st.spinner("Processing community detection..."):
                community_detection_and_visualization(graph, similarity_threshold=0.98)



def create_footer() -> None:
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
    db_path, html_path = get_args()
    st.set_page_config(page_title="MDdatasetExplorer", layout="wide", page_icon="⚛", initial_sidebar_state='collapsed')
    create_banner()
    create_sidebar()
    create_main_page(db_path, html_path)
    create_footer()