import streamlit as st
import pandas as pd
def display_data_information(dataframe):
    st.subheader(":orange[Data Information]", anchor='center')
    st.write("Number of Rows:", dataframe.shape[0])
    st.write("Number of Columns:", dataframe.shape[1])
    st.write("Data Types:", dataframe.dtypes)

# Function to display missing values
def display_missing_values(dataframe):
    st.subheader(":orange[Missing Values]", anchor='center')
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
    st.subheader(":orange[Descriptive Statistics]", anchor='center')
    st.write(dataframe.describe())