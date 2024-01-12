import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def delete_columns(dataframe):
    '''
    Delete selected columns from the DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame after deleting selected columns.

    Purpose: Delete selected columns from the DataFrame.
    Role: Utilizes Streamlit forms for user input and removes selected columns.
    Usage in the Application: Used in the "Upload Data" tab for data preprocessing.
    '''
    with st.form(key='delete_columns_form'):
        columns_to_delete = st.multiselect("Select columns to delete", dataframe.columns.tolist())
        enable_deletion = st.form_submit_button("Delete")

        if enable_deletion:
            # Remove selected columns from the DataFrame
            dataframe = dataframe.drop(columns=columns_to_delete, errors='ignore')

    return dataframe


def handle_missing_values_numeric(df):
    '''
    Handle missing values in numeric columns using selected techniques.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame after handling missing values.

    Purpose: Handle missing values in numeric columns using selected techniques.
    Role: Provides options for imputation or deletion of missing values in numeric columns.
    Usage in the Application: Used in the "Visualize Data" and "Data Modeling" tabs for data preprocessing.
    '''
    with st.sidebar.form(key='missing_values_numeric_form'):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        missing_values_option = st.radio("Technique", ["None", "Imputation", "Deletion"])
        submit_button = st.form_submit_button("Apply")

        if submit_button:
            if missing_values_option == "Imputation":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif missing_values_option == "Deletion":
                df = df.dropna(subset=numeric_cols)


    return df

def handle_missing_values_categorical(df):
    '''
    Handle missing values in categorical columns using selected techniques.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame after handling missing values.

    Purpose: Handle missing values in categorical columns using selected techniques.
    Role: Offers choices for imputing with the most frequent value or creating a new category.
    Usage in the Application: Employed in the "Visualize Data" and "Data Modeling" tabs for data preprocessing.
    '''
    with st.sidebar.form(key='missing_values_categorical_form'):
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        missing_values_option_cat = st.radio("Technique", ["None", "Most Frequent", "New Category"])
        submit_button = st.form_submit_button("Apply")

        if submit_button:
            if missing_values_option_cat == "Most Frequent":
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            elif missing_values_option_cat == "New Category":
                df[categorical_cols] = df[categorical_cols].fillna("Missing")


    return df

def handle_outliers(df, threshold):
    '''
    Handle outliers in numeric columns using selected techniques.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Z-score threshold for outlier detection.

    Returns:
        pd.DataFrame: DataFrame after handling outliers.

Purpose: Handle outliers in numeric columns using selected techniques.
Role: Options for handling outliers include Z-score-based removal or IQR-based removal.
Usage in the Application: Used in the "Visualize Data" and "Data Modeling" tabs for data preprocessing
    '''
    with st.sidebar.form(key='outliers_form'):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        outlier_method = st.radio("Outlier Handling Method", ["Z-score", "IQR"])  # Add radio button for outlier method
        submit_button = st.form_submit_button("Apply")

        if submit_button:
            if outlier_method == "Z-score":
                threshold = st.number_input("Z-score Threshold", value=threshold)
                z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
                df = df[(z_scores < threshold).all(axis=1)]
            elif outlier_method == "IQR":
                q1 = df[numeric_cols].quantile(0.25)
                q3 = df[numeric_cols].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df = df[~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)]

        return df

def encoding_categorical(dataframe):
    '''
    Encode categorical columns using selected techniques.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame after encoding categorical columns.

    Purpose: Encode categorical columns using selected techniques.
    Role: Provides options for label encoding or one-hot encoding based on user choices.
    Usage in the Application: Employed in the "Visualize Data" and "Data Modeling" tabs for data preprocessing.
    '''
    with st.sidebar.form(key='encoding_form'):
        all_categorical_cols = dataframe.select_dtypes(include='object').columns.tolist()
        cols_to_exclude = st.multiselect("Select columns to exclude from encoding", all_categorical_cols)
        cols_to_encode = list(set(all_categorical_cols) - set(cols_to_exclude))
        encoding_categorical_option_cat = st.radio("Technique", ["Label Encoding", "One Hot Encoding"])
        submit_encoding = st.form_submit_button("Apply Encoding")

    if submit_encoding:
        if encoding_categorical_option_cat == "Label Encoding":
            for col in cols_to_encode:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].astype('category').cat.codes
        elif encoding_categorical_option_cat == "One Hot Encoding":
            dataframe = pd.get_dummies(dataframe, columns=cols_to_encode)

    return dataframe


def scaler(dataframe):
    '''
    Scale numerical columns using selected techniques.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame after scaling numerical columns.

    Purpose: Scale numerical columns using selected techniques.
    Role: Options include normalization, standardization, or Min-Max scaling.
    Usage in the Application: Used in the "Visualize Data" and "Data Modeling" tabs for data preprocessing.
    '''
    with st.sidebar.form(key='scaling_form'):
        feature_scaling_option = st.radio("Technique", ["None", "Normalization", "Standardization", "MinMaxScaler"])
        submit_scaling = st.form_submit_button("Apply Scaling")

    if submit_scaling:
        # Get numerical columns
        numeric_cols = dataframe.select_dtypes(include=np.number).columns.tolist()

        # Apply scaling based on user choices
        if feature_scaling_option == "Normalization":
            scaler = MinMaxScaler()
            dataframe[numeric_cols] = scaler.fit_transform(dataframe[numeric_cols])
        elif feature_scaling_option == "Standardization":
            scaler = StandardScaler()
            dataframe[numeric_cols] = scaler.fit_transform(dataframe[numeric_cols])
        elif feature_scaling_option == "MinMaxScaler":
            scaler = MinMaxScaler(feature_range=(-1, 1))
            dataframe[numeric_cols] = scaler.fit_transform(dataframe[numeric_cols])

    return dataframe