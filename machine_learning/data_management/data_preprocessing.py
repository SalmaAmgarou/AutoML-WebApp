import streamlit as st

def select_target(dataframe):
    if dataframe is None:
        return None

    st.sidebar.header('Select column to predict')
    target = tuple(dataframe.columns)
    selected_target = st.sidebar.selectbox('Select target for prediction', target)

    confirm_button = st.sidebar.button("Confirm Target")

    if confirm_button:
        st.sidebar.write('Selected target:', selected_target)
        preprocessing_options(dataframe)

def preprocessing_options(dataframe):
    st.sidebar.subheader("Preprocessing Options")

    missing_values_option = st.sidebar.radio("Handle Missing Values", ["None", "Imputation", "Deletion"])
    encoding_option = st.sidebar.radio("Encoding Categorical Values", ["None", "One Hot Encoding", "Label Encoding"])
    outliers_option = st.sidebar.radio("Handle Outliers", ["None", "Z-Score", "IQR"])
    feature_scaling_option = st.sidebar.radio("Select Scaling Methods", ["None", "Normalization", "Standardization"])
