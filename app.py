import pandas as pd
import streamlit as st
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance


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


def read_csv_or_excel(file):
    # Read CSV or Excel file
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

def find_exact_match(df1, df2, column_name):
    # Find rows with exact matches in the specified column
    matches = pd.merge(df1, df2, on=column_name, how='inner')
    return matches


def find_similar_texts(df1, df2, column_name, threshold=0.3):
    # Find rows with similar texts in the specified column, excluding exact matches
    similar_texts = []
    exact_matches = []

    # Convert numeric values to strings
    df1[column_name] = df1[column_name].astype(str)
    df2[column_name] = df2[column_name].astype(str)
    
    # Concatenate texts from both dataframes
    all_texts = df1[column_name].tolist() + df2[column_name].tolist()
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Iterate over pairs of rows to find similar texts
    for i, row1 in df1.iterrows():
        for j, row2 in df2.iterrows():
            similarity = similarity_matrix[i, len(df1) + j]
            if similarity >= threshold:
                # Calculate Levenshtein distance between strings
                distance = levenshtein_distance(row1[column_name], row2[column_name])
                max_length = max(len(row1[column_name]), len(row2[column_name]))
                similarity_score = 1 - (distance / max_length)
                if similarity_score >= threshold:
                    if similarity == 1:  # Exact match
                        exact_matches.append((i, j, row1[column_name], row2[column_name]))
                    elif similarity < 0.99:  # Similar but not the same
                        similar_texts.append((i, j, row1[column_name], row2[column_name]))
    
    return similar_texts, exact_matches



def main():
    st.title("Item Comparison App")

    # Upload files
    st.header("Upload Files")
    warehouse_file = st.file_uploader("Upload Warehouse Item Stocks (CSV or Excel)")
    industry_file = st.file_uploader("Upload Industry Item Stocks (CSV or Excel)")

    if warehouse_file is not None and industry_file is not None:
        # Read files
        warehouse_df = read_csv_or_excel(warehouse_file)
        industry_df = read_csv_or_excel(industry_file)

        # Get column names
        warehouse_columns = warehouse_df.columns.tolist()
        industry_columns = industry_df.columns.tolist()

        # Select columns using dropdowns
        st.header("Select Columns")
        warehouse_column = st.selectbox("Choose column from warehouse item stocks:", warehouse_columns)
        industry_column = st.selectbox("Choose column from industry item stocks:", industry_columns)
 
        # Compare button
        if st.button("Compare"):
            # Find exact matches
            exact_match = find_exact_match(warehouse_df, industry_df, warehouse_column)

            # Find similar texts
            similar_texts, exact_matches = find_similar_texts(warehouse_df, industry_df, warehouse_column)

            # Display results
            st.header("Exact Matches")
            st.write(exact_match)
 

           # Display exact matches
            st.header("Exact Matches Compare")
            for match in exact_matches:
                st.write(f"Row {match[0]} in warehouse item stocks is exactly the same as Row {match[1]} in industry item stocks:")
                st.write(f"Warehouse: {match[2]}")
                st.write(f"Industry: {match[3]}")
                st.write(f"____________________")
                st.write()

            # Display similar texts
            st.header("Similar (but Not Same) Texts")
            for text_pair in similar_texts:
                st.write(f"Row {text_pair[0]} in warehouse item stocks is similar to Row {text_pair[1]} in industry item stocks:")
                st.write(f"Warehouse: {text_pair[2]}")
                st.write(f"Industry: {text_pair[3]}")
                st.write(f"____________________")
                st.write()


if __name__ == "__main__":
    main()
