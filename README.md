# Same-and-Similar-Extractor-App

# Item Comparison App

This Streamlit web application allows users to compare item stocks between a warehouse and an industry dataset. Users can upload CSV or Excel files containing item stocks for both warehouses and industries, select the columns to compare, and click the "Compare" button to find exact matches and similar (but not the same) texts.

## Features

- **Upload Files**: Users can upload CSV or Excel files containing item stocks for warehouses and industries.
- **Select Columns**: Users can choose the columns from both datasets that they want to compare.
- **Compare**: After selecting the columns, users can click the "Compare" button to find exact matches and similar texts.
- **Exact Matches**: Displays rows where the items are exactly the same in both datasets.
- **Similar (but Not Same) Texts**: Displays rows where the items are similar but not identical in both datasets.

## Usage

1. **Upload Files**: Click on the "Upload Files" section and upload CSV or Excel files containing item stocks for the warehouse and industry.

2. **Select Columns**: Choose the columns from both datasets that you want to compare using the dropdowns provided.

3. **Compare**: Click the "Compare" button to find exact matches and similar texts between the selected columns.

4. **View Results**: Explore the results in the "Exact Matches" and "Similar (but Not Same) Texts" sections, where each row is displayed along with the corresponding item details from the warehouse and industry datasets.

## Requirements

- pandas
- streamlit
- scikit-learn
- python-Levenshtein

## Installation

1. Clone the repository:

```
git clone <repository-url>
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the Streamlit app:

```
streamlit run app.py
```

## About

This application was created to facilitate the comparison of item stocks between warehouses and industries. It utilizes text similarity techniques such as TF-IDF and Levenshtein distance to identify exact matches and similar items between datasets.

