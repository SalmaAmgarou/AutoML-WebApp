import streamlit as st

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
        handle_missing_values(dataframe)
        encoding_categorical(dataframe)
        outliers(dataframe)
        scaler(dataframe)


def handle_missing_values(dataframe):
    with st.sidebar.form(key='missing_values_form'):
        st.write("Handle Missing Values")
        missing_values_all = st.checkbox("Apply to all columns")
        missing_values_spec = st.multiselect("Apply to specific columns", dataframe.columns.tolist())
        missing_values_option = st.radio("Technique", ["None", "Imputation", "Deletion"])
        submit_missing_values = st.form_submit_button("Apply Missing Values")
    return dataframe


def encoding_categorical(dataframe):
    with st.sidebar.form(key='encoding_form'):
        st.write("Encoding Categorical Values")
        encoding_all = st.checkbox("Apply to all columns")
        encoding_spec = st.multiselect("Apply to specific columns", dataframe.select_dtypes(include='object').columns.tolist())
        encoding_option = st.radio("Technique", ["None", "One Hot Encoding", "Label Encoding"])
        submit_encoding = st.form_submit_button("Apply Encoding")
    return dataframe


def outliers(dataframe):
    with st.sidebar.form(key='outliers_form'):
        st.write("Handle Outliers")
        outliers_all = st.checkbox("Apply to all columns")
        outliers_spec = st.multiselect("Apply to specific columns", dataframe.select_dtypes(exclude='object').columns.tolist())
        outliers_option = st.radio("Technique", ["None", "Z-Score", "IQR"])
        submit_outliers = st.form_submit_button("Apply Outliers")

    return dataframe


def scaler(dataframe):
    with st.sidebar.form(key='scaling_form'):
        st.write("Select Scaling Methods")
        scaling_all = st.checkbox("Apply to all columns")
        scaling_spec = st.multiselect("Apply to specific columns", dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist())
        feature_scaling_option = st.radio("Technique", ["None", "Normalization", "Standardization"])
        submit_scaling = st.form_submit_button("Apply Scaling")
    return dataframe











