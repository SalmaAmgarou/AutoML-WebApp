import streamlit as st
from user_interface.window_main import display_data_information, display_missing_values, display_descriptive_statistics
from machine_learning.data_management.data_loader import load_data
from machine_learning.data_management.data_preprocessing import (
    handle_missing_values_numeric,
    handle_outliers,
    handle_missing_values_categorical,
    encoding_categorical,
    scaler,
)
import time
from machine_learning.data_management.Visualization import visualize_data

def main():
    st.set_page_config(page_title="Machine Learning in Action", page_icon="💾")
    col1, col2 = st.columns([1, 1])  # Two columns of equal width
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    .dot-matrix {
        font-family: 'VT323', monospace;
        font-size: 32px;
        margin: 0; /* Set margins to zero */
        padding: 0; /* Set padding to zero */
    }
    .title {
        font-family: 'VT323', monospace;
        font-size: 60px;
    }
    .titles {
        font-family: 'VT323', monospace;
        font-size: 32px;
    }
    .titles {
        font-family: 'VT323', monospace;
        font-size: 30px;
        color: #FFA500;
    }
    .visualize{
        font-family: 'VT323', monospace;
        font-size: 38px;
    }
    .divider{
        font-family: 'VT323', monospace;
        font-size: 32px;
        margin: 0; /* Set margins to zero */
        padding: 0; /* Set padding to zero */
        color:#000000;
    }
    </style>
    """, unsafe_allow_html=True)
    with col1:
        st.image('images/LOGO IADS FR.png',  width=240)
    with col2:
        st.image('images/fstt.png',  width=420)

    st.markdown('<p class="title">Machine Learning In Action</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Upload Data", "Visualize Data"])

    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'preprocessed_df' not in st.session_state:
        st.session_state.preprocessed_df = None

    with tab1:
        with st.status("Downloading data..."):
            time.sleep(0.8)
            st.session_state.original_df = load_data()


        if st.session_state.original_df is not None:
            st.success('Data Loaded Successfully!')
            st.write(":red[Switch to the 'Visualize Data' tab to see the visualizations.]")
            columns = st.columns(2)
            if st.session_state.preprocessed_df is None:
                preprocessed_df = st.session_state.original_df.copy()
            else:
                preprocessed_df = st.session_state.preprocessed_df.copy()
            st.sidebar.markdown('<p class="divider">#########################</p>', unsafe_allow_html=True)
            st.sidebar.markdown('<p class="dot-matrix">Features selection</p>', unsafe_allow_html=True)
            st.sidebar.markdown('<p class="divider">#########################</p>', unsafe_allow_html=True)
            missing_values_numeric = st.sidebar.checkbox("Handle Missing Values (Numeric)")
            if missing_values_numeric:
                preprocessed_df = handle_missing_values_numeric(preprocessed_df)
            missing_values_categorical = st.sidebar.checkbox("Handle Missing Values (Categorical)")
            if missing_values_categorical:
                preprocessed_df = handle_missing_values_categorical(preprocessed_df)
            handle_out = st.sidebar.checkbox("Handle Outliers (Z-score)")
            if handle_out:
                preprocessed_df = handle_outliers(preprocessed_df, threshold=3.0)
            st.sidebar.markdown('<p class="divider">#########################</p>', unsafe_allow_html=True)
            st.sidebar.markdown('<p class="dot-matrix">Data transformation</p>', unsafe_allow_html=True)
            st.sidebar.markdown('<p class="divider">#########################</p>', unsafe_allow_html=True)
            encoding = st.sidebar.checkbox("Encoding categorical and numeric")
            if encoding:
                preprocessed_df = encoding_categorical(preprocessed_df)
            scaling = st.sidebar.checkbox("Scaling and normalizing")
            if scaling:
                preprocessed_df = scaler(preprocessed_df)

            st.header("")
            st.data_editor(preprocessed_df)
            st.header("")
            col2, col3 = st.columns([1, 1])
            with col2:
                display_data_information(preprocessed_df)
            with col3:
                display_missing_values(preprocessed_df)
            display_descriptive_statistics(preprocessed_df)

            st.session_state.preprocessed_df = preprocessed_df

    with tab2:
        visualize_data(st.session_state.preprocessed_df)

if __name__ == "__main__":
    main()

