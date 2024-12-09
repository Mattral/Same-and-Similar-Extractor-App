import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import os

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

            # Display results
            st.header("Exact Matches")
            st.write(exact_match)

            # Display exact matches
            st.header("Exact Matches Compare")
            for match in exact_match.itertuples():
                st.write(f"Row {match.Index + 2} in File-1 item stocks is exactly the same as Row {match.Index + 2} in File-2 item stocks:")
                st.write(f"Warehouse: {match[1]}")
                st.write(f"Industry: {match[2]}")
                st.write(f"____________________")
                st.write()


if __name__ == "__main__":
    main()
