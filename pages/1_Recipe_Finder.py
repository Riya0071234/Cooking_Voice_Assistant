# pages/1_Recipe_Finder.py
"""
A UI page to browse and search for recipes stored in the database.
"""

import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/recipes"

st.set_page_config(page_title="Recipe Finder", page_icon="📖")
st.title("📖 Recipe Finder")
st.markdown("Search our database for recipes.")

# --- Search and Filter ---
search_term = st.text_input("Search by keyword (e.g., 'chicken', 'pasta')")
if st.button("Search"):
    try:
        response = requests.get(API_URL, params={"search": search_term}, timeout=30)
        response.raise_for_status()
        recipes = response.json()

        if not recipes:
            st.warning("No recipes found matching your search term.")
        else:
            for recipe in recipes:
                with st.expander(f"{recipe['title']} ({recipe.get('cuisine', 'N/A')})"):
                    st.subheader("Ingredients")
                    for ingredient in recipe['ingredients']:
                        st.markdown(f"- {ingredient}")

                    st.subheader("Instructions")
                    for i, instruction in enumerate(recipe['instructions']):
                        st.markdown(f"{i + 1}. {instruction}")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the recipe database. Please ensure the backend is running. Error: {e}")