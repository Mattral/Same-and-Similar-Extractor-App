import pandas as pd
import streamlit as st
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import os

ms = st.session_state
if "themes" not in ms: 
  ms.themes = {"current_theme": "light",
                    "refreshed": True,
                    
                    "light": {"theme.base": "dark",
                              "theme.backgroundColor": "black",
                              "theme.primaryColor": "#c98bdb",
                              "theme.secondaryBackgroundColor": "#5591f5",
                              "theme.textColor": "white",
                              "theme.textColor": "white",
                              "button_face": "🌜"},

                    "dark":  {"theme.base": "light",
                              "theme.backgroundColor": "white",
                              "theme.primaryColor": "#5591f5",
                              "theme.secondaryBackgroundColor": "#82E1D7",
                              "theme.textColor": "#0a1464",
                              "button_face": "🌞"},
                    }
  

def ChangeTheme():
  previous_theme = ms.themes["current_theme"]
  tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
  for vkey, vval in tdict.items(): 
    if vkey.startswith("theme"): st._config.set_option(vkey, vval)

  ms.themes["refreshed"] = False
  if previous_theme == "dark": ms.themes["current_theme"] = "light"
  elif previous_theme == "light": ms.themes["current_theme"] = "dark"


btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
st.button(btn_face, on_click=ChangeTheme)

if ms.themes["refreshed"] == False:
  ms.themes["refreshed"] = True
  st.rerun()


def read_csv_or_excel_and_convert(file):
    """Read CSV or Excel file and save as Parquet for faster subsequent access, read in chunks."""
    file_path = file.name
    parquet_path = file_path + '.parquet'

    if file_path.endswith('.csv'):
        df = pd.read_csv(file, dtype={"float64": "float32", "int64": "int32"})
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file, dtype={"float64": "float32", "int64": "int32"})
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

    # Save to Parquet
    df.to_parquet(parquet_path)

    return df, parquet_path

def read_parquet_file(parquet_path):
    """Read data from a Parquet file."""
    return pd.read_parquet(parquet_path)
      

def find_exact_match(df1, df2, column_name, chunk_size=1500):
    # Ensure the column for merging has the same data type and is cleaned
    df1[column_name] = df1[column_name].astype(str).str.strip()
    df2[column_name] = df2[column_name].astype(str).str.strip()

    # Initialize an empty DataFrame to store matches
    matches = pd.DataFrame()

    # Process in chunks
    num_chunks = (len(df1) + chunk_size - 1) // chunk_size  # Calculate number of chunks

    for i in range(num_chunks):
        chunk = df1.iloc[i*chunk_size:(i+1)*chunk_size]
        matched_chunk = pd.merge(chunk, df2, on=column_name, how='inner')
        matches = pd.concat([matches, matched_chunk], ignore_index=True)

    return matches


def find_similar_texts(df1, df2, column_name, threshold=0.3):
    similar_texts = []
    exact_matches = []

    df1[column_name] = df1[column_name].astype(str)
    df2[column_name] = df2[column_name].astype(str)

    all_texts = df1[column_name].tolist() + df2[column_name].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    chunk_size = 1500
    for start in range(0, len(df1), chunk_size):
        end = start + chunk_size
        chunk1 = df1.iloc[start:end].reset_index(drop=True)

        for start2 in range(0, len(df2), chunk_size):
            end2 = start2 + chunk_size
            chunk2 = df2.iloc[start2:end2].reset_index(drop=True)

            for i, row1 in chunk1.iterrows():
                for j, row2 in chunk2.iterrows():
                    similarity = similarity_matrix[start + i, len(df1) + start2 + j]
                    if similarity >= threshold:
                        distance = levenshtein_distance(row1[column_name], row2[column_name])
                        max_length = max(len(row1[column_name]), len(row2[column_name]))
                        similarity_score = 1 - (distance / max_length)
                        if similarity_score >= threshold:
                            if similarity == 1:  # Exact match
                                exact_matches.append((start + i, start2 + j, row1[column_name], row2[column_name]))
                            elif similarity < 0.99:  # Similar but not the same
                                similar_texts.append((start + i, start2 + j, row1[column_name], row2[column_name]))

    return similar_texts, exact_matches

def main():
    st.title("Item Comparison App")

    # Upload files
    st.header("Upload Files")
    warehouse_file = st.file_uploader("Upload File-1 Items (CSV or Excel)", key='warehouse')
    industry_file = st.file_uploader("Upload File-2 Items (CSV or Excel)", key='industry')

    if warehouse_file is not None and industry_file is not None:
        # Read files and convert to Parquet
        warehouse_df, warehouse_parquet = read_csv_or_excel_and_convert(warehouse_file)
        industry_df, industry_parquet = read_csv_or_excel_and_convert(industry_file)

        # Get column names from Parquet (optionally could use original DFs)
        warehouse_columns = warehouse_df.columns.tolist()
        industry_columns = industry_df.columns.tolist()

        # Column selection interface
        st.header("Select Columns")
        warehouse_column = st.selectbox("Choose column from File-1 item stocks:", warehouse_columns)
        industry_column = st.selectbox("Choose column from File-2 item stocks:", industry_columns)

        # Compare button
        if st.button("Compare"):
            # Read from Parquet for faster loading
            warehouse_df = read_parquet_file(warehouse_parquet)
            industry_df = read_parquet_file(industry_parquet)

            # Find exact matches
            exact_match = find_exact_match(warehouse_df, industry_df, warehouse_column)

            # Find similar texts
            similar_texts, exact_matches = find_similar_texts(warehouse_df, industry_df, warehouse_column)

            # Display results
            st.header("Exact Matches")
            st.write(exact_match)
 


if __name__ == "__main__":
    main()
