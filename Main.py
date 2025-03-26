import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load your final measurement dataset
df = pd.read_csv("Final_Measurement_Framework.csv")

# Streamlit UI setup
st.set_page_config(page_title="Measurement Finder", layout="wide")
st.title("📊 Measurement Selector Based on Your Objectives")

# Model selection dropdown
model_choice = st.selectbox("Choose an embedding model:", [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
])

# Load selected sentence transformer model
model = SentenceTransformer(model_choice)

# Combine Business + Campaign Objective for semantic matching
df["combined_objective"] = df["Business Objective"] + ": " + df["Campaign Objective"]
unique_rows = df.drop_duplicates(subset="combined_objective")
df_embeddings = model.encode(unique_rows["combined_objective"].tolist(), convert_to_tensor=True).cpu().numpy()

# User input
biz_obj = st.text_input("Describe your **business objective**")
camp_obj = st.text_input("Describe your **campaign objective**")

# Run matching if inputs are provided
if biz_obj and camp_obj:
    user_input = f"{biz_obj}: {camp_obj}"
    user_embedding = model.encode([user_input], convert_to_tensor=True).cpu().numpy()

    # Compute cosine similarity to each unique objective
    scores = cosine_similarity(user_embedding, df_embeddings)[0]
    unique_rows["similarity"] = scores

    # Merge similarity scores back into full dataframe
    df = df.merge(unique_rows[["combined_objective", "similarity"]], on="combined_objective", how="left")

    # Highlight all relevant measurement methods based on matching objectives
    relevant_methods = df[df["similarity"] > 0.6]["Measurement Method"].unique().tolist()

    st.markdown("---")
    st.subheader("🧪 Recommended Measurements")

    # Create tile layout (4 columns per row)
    cols = st.columns(4)
    for i, method in enumerate(sorted(df["Measurement Method"].unique())):
        col = cols[i % 4]
        is_highlighted = method in relevant_methods
        button_style = "background-color:#d0f0c0;" if is_highlighted else "background-color:#f0f0f0;"

        with col:
            if st.button(method, key=method, help="Click for details"):
                selected_row = df[df["Measurement Method"] == method].iloc[0]
                st.markdown("---")
                st.markdown(f"### {method}")
                st.markdown(f"**Description:** {selected_row['Description']}")
                st.markdown(f"**Implementation Cost:** {selected_row['Implementation Cost (1=Low, 5=High)']}")
                st.markdown(f"**Impact Duration:** {selected_row['Impact Duration (1=Short, 5=Long)']}")
