import streamlit as st
import pandas as pd
def display_data_information(dataframe):
    st.header("")
    st.markdown('<p class="titles">Data Information</p>', unsafe_allow_html=True)
    st.write("Number of Rows:", dataframe.shape[0])
    st.write("Number of Columns:", dataframe.shape[1])
    st.write("Data Types:", dataframe.dtypes)

# Function to display missing values
def display_missing_values(dataframe):
    st.header("")
    st.markdown('<p class="titles">Missing Values</p>', unsafe_allow_html=True)
    missing_values = dataframe.isnull().sum()
    columns_with_missing_values = missing_values[missing_values > 0]

    if columns_with_missing_values.empty:
        st.write("No missing values found in columns.")
    else:
        st.write("Columns with missing values:")
        for column, count in columns_with_missing_values.items():
            st.text(f"Column: '{column}' - Missing values: {count}")
    st.write(missing_values)

# Function to display descriptive statistics
def display_descriptive_statistics(dataframe):
    st.header("")
    st.markdown('<p class="titles">Descriptive Statistics</p>', unsafe_allow_html=True)
    st.header("")
    st.write(dataframe.describe())