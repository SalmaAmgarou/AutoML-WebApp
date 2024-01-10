import streamlit as st
import pandas as pd
import numpy as np
def select_target(dataframe):
    if dataframe is None:
        return None

    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None

    st.sidebar.header('Select column to predict')
    target = tuple(dataframe.columns)
    selected_target = st.sidebar.selectbox('Select target for prediction', target)

    confirm_button = st.sidebar.button("Confirm Target")

    if confirm_button:
        st.sidebar.write('Selected target:', selected_target)
        st.session_state.selected_target = selected_target

def handle_missing_values_numeric(df):
    with st.sidebar.form(key='missing_values_numeric_form'):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        missing_values_option = st.radio("Technique", ["None", "Imputation", "Deletion"])

        if missing_values_option != "None":
            if missing_values_option == "Imputation":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif missing_values_option == "Deletion":
                df = df.dropna(subset=numeric_cols)

        submit_button = st.form_submit_button("Apply")

    return df

def handle_missing_values_categorical(df):
    with st.sidebar.form(key='missing_values_categorical_form'):
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        missing_values_option_cat = st.radio("Technique", ["None", "Most Frequent", "New Category"])

        if missing_values_option_cat != "None":
            if missing_values_option_cat == "Most Frequent":
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            elif missing_values_option_cat == "New Category":
                df[categorical_cols] = df[categorical_cols].fillna("Missing")

        submit_button = st.form_submit_button("Apply")

    return df

def handle_outliers(df, threshold):
    with st.sidebar.form(key='outliers_form'):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        outlier_method = st.radio("Outlier Handling Method", ["Z-score", "IQR"])  # Add radio button for outlier method

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

        submit_button = st.form_submit_button("Apply")
    return df

def encoding_categorical(dataframe):
    with st.sidebar.form(key='encoding_form'):
        encoding_categorical_option_cat = st.radio("Technique", ["None", "One Hot Encoding", "Label Encoding"])
        submit_scaling = st.form_submit_button("Apply Scaling")
    return dataframe



def scaler(dataframe):
    with st.sidebar.form(key='scaling_form'):
        feature_scaling_option = st.radio("Technique", ["None", "Normalization", "Standardization"])
        submit_scaling = st.form_submit_button("Apply Scaling")
    return dataframe











