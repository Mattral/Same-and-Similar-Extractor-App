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
      

def find_exact_match(df1, df2, column_name, chunk_size=500):
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



def find_similar_texts(df1, df2, column_name, threshold=0.3, chunk_size=500):
    similar_texts = []
    exact_matches = []

    # Ensure the column for matching is of type string
    df1[column_name] = df1[column_name].astype(str)
    df2[column_name] = df2[column_name].astype(str)

    # Define a function to process chunks
    def process_chunk(chunk1, chunk2):
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer()
        all_texts = chunk1.tolist() + chunk2.tolist()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Check pairs
        for i in range(len(chunk1)):
            for j in range(len(chunk2)):
                index2 = j + len(chunk1)
                similarity = similarity_matrix[i, index2]
                if similarity >= threshold:
                    distance = levenshtein_distance(chunk1[i], chunk2[j])
                    max_length = max(len(chunk1[i]), len(chunk2[j]))
                    similarity_score = 1 - (distance / max_length)
                    if similarity_score >= threshold:
                        if similarity == 1:
                            exact_matches.append((chunk1.index[i], chunk2.index[j], chunk1[i], chunk2[j]))
                        else:
                            similar_texts.append((chunk1.index[i], chunk2.index[j], chunk1[i], chunk2[j]))

    # Iterate over chunks
    num_chunks_df1 = (len(df1) + chunk_size - 1) // chunk_size  # Calculate number of chunks
    num_chunks_df2 = (len(df2) + chunk_size - 1) // chunk_size

    for i in range(num_chunks_df1):
        chunk1 = df1[column_name].iloc[i*chunk_size:(i+1)*chunk_size]
        for j in range(num_chunks_df2):
            chunk2 = df2[column_name].iloc[j*chunk_size:(j+1)*chunk_size]
            process_chunk(chunk1, chunk2)

    return similar_texts, exact_matches

def plot_correlation(df, column):
    plt.figure(figsize=(8, 6))
    plt.scatter(df.index, df[column])
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.title(f"Correlation Plot of {column}")
    return plt.gcf()  # Return the matplotlib figure

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_correlation_matrix(df):
    # Filter for numeric columns, if the DataFrame has non-numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
    st.pyplot()  # Use Streamlit's method to display the plot

def main():
    st.title("Item Comparison App")

    # Upload files
    st.header("Upload Files")
    warehouse_file = st.file_uploader("Upload Warehouse Item Stocks (CSV or Excel)", key='warehouse')
    industry_file = st.file_uploader("Upload Industry Item Stocks (CSV or Excel)", key='industry')

    if warehouse_file is not None and industry_file is not None:
        # Read files and convert to Parquet
        warehouse_df, warehouse_parquet = read_csv_or_excel_and_convert(warehouse_file)
        industry_df, industry_parquet = read_csv_or_excel_and_convert(industry_file)

        # Get column names from Parquet (optionally could use original DFs)
        warehouse_columns = warehouse_df.columns.tolist()
        industry_columns = industry_df.columns.tolist()

        # Column selection interface
        st.header("Select Columns")
        warehouse_column = st.selectbox("Choose column from warehouse item stocks:", warehouse_columns)
        industry_column = st.selectbox("Choose column from industry item stocks:", industry_columns)

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
 

           # Display exact matches
            st.header("Exact Matches Compare")
            for match in exact_matches:
                st.write(f"Row {match[0]+2} in warehouse item stocks is exactly the same as Row {match[1]+2} in industry item stocks:")
                st.write(f"Warehouse: {match[2]}")
                st.write(f"Industry: {match[3]}")
                st.write(f"____________________")
                st.write()

            # Display similar texts
            st.header("Similar (but Not Same) Texts")
            for text_pair in similar_texts:
                st.write(f"Row {text_pair[0]+2} in warehouse item stocks is similar to Row {text_pair[1]+2} in industry item stocks:")
                st.write(f"Warehouse: {text_pair[2]}")
                st.write(f"Industry: {text_pair[3]}")
                st.write(f"____________________")
                st.write()

            if warehouse_df[warehouse_column].dtype != "object" and industry_df[industry_column].dtype != "object":

                # Calculate correlation
                correlation = warehouse_df[warehouse_column].corr(industry_df[industry_column])
                st.header("Correlation")
                st.write(f"The correlation between {warehouse_column} in warehouse item stocks and {industry_column} in industry item stocks is: {correlation}")
                st.write()


        # Show correlation plot for each dataset
        if st.button("Correlation for each dataset"):
            
            st.subheader("Correlation Plot for 1st Dataset")
            warehouse_corr_plot = plot_correlation(warehouse_df, warehouse_column)
            st.pyplot(warehouse_corr_plot)
                
            st.subheader("Correlation Plot for 2nd Dataset")
            industry_corr_plot = plot_correlation(industry_df, industry_column)
            st.pyplot(industry_corr_plot)

            st.subheader("Correlation Matrix for 1st Dataset")
            plot_correlation_matrix(warehouse_df)

            st.subheader("Correlation Matrix for 2nd Dataset")
            plot_correlation_matrix(industry_df)

if __name__ == "__main__":
    main()
