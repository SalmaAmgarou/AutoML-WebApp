import streamlit as st
import pandas as pd

# Function to display data information
def display_data_information(dataframe):
    # Displaying a header
    st.header("")
    # Using markdown for styling the title
    st.markdown('<p class="titles">Data Information</p>', unsafe_allow_html=True)
    # Displaying the number of rows, columns, and data types of the dataframe
    st.write("Number of Rows:", dataframe.shape[0])
    st.write("Number of Columns:", dataframe.shape[1])
    st.write("Data Types:", dataframe.dtypes)

# Function to display missing values
def display_missing_values(dataframe):
    # Displaying a header
    st.header("")
    # Using markdown for styling the title
    st.markdown('<p class="titles">Missing Values</p>', unsafe_allow_html=True)
    # Calculating missing values for each colum
    missing_values = dataframe.isnull().sum()
    # Filtering columns with missing values
    columns_with_missing_values = missing_values[missing_values > 0]

    # Checking if there are any missing values
    if columns_with_missing_values.empty:
        st.write("No missing values found in columns.")
    else:
        st.write("Columns with missing values:")
        # Displaying columns with missing values and their respective counts
        for column, count in columns_with_missing_values.items():
            st.text(f"Column: '{column}' - Missing values: {count}")
    # Displaying the overall missing values count for each column
    st.write(missing_values)

# Function to display descriptive statistics
def display_descriptive_statistics(dataframe):
    st.header("")
    st.markdown('<p class="titles">Descriptive Statistics</p>', unsafe_allow_html=True)
    st.header("")
    # Displaying the descriptive statistics of the dataframe
    st.write(dataframe.describe())