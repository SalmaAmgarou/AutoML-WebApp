import streamlit as st
from user_interface.window_main import display_data_information, display_missing_values, display_descriptive_statistics
from machine_learning.data_management.data_loader import load_data
from machine_learning.data_management.data_preprocessing import (
    handle_missing_values_numeric,
    handle_outliers_zscore,
    handle_missing_values_categorical,
    encoding_categorical,
    scaler,
)
from machine_learning.data_management.Visualization import visualize_data
import time

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
        font-size: 38px;
        margin: 0; /* Set margins to zero */
        padding: 0; /* Set padding to zero */
    }
    </style>
    """, unsafe_allow_html=True)
    with col1:
        st.image('images/LOGO IADS FR.png',  width=240)
    with col2:
        st.image('images/fstt.png',  width=420)

    st.markdown('<p class="title">Machine Learning In Action</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Upload Data", "Visualize Data"])
    df = None
    with tab1:
        with st.status("Downloading data..."):
            time.sleep(0.8)
            df = load_data()

        if df is not None:
            st.success('Data Downloaded Successfully!', icon="✅")
            st.write(":red[Switch to the 'Visualize Data' tab to see the visualizations.]")
            columns = st.columns(2)

            with columns[0]:
                st.sidebar.markdown('<p class="divider">------------------------</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="dot-matrix">Features selection</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="divider">------------------------</p>', unsafe_allow_html=True)
                st.sidebar.markdown(":red[Handle Missing Values (Numeric)]")
                df = handle_missing_values_numeric(df)
                st.sidebar.markdown(":red[Handle Missing Values (Categorical)]")
                df = handle_missing_values_categorical(df)
                st.sidebar.markdown(":red[Handle Outliers (Z-score) (Numeric)]")
                df = handle_outliers_zscore(df, threshold=3.0)
                st.sidebar.markdown('<p class="divider">------------------------</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="dot-matrix">Data transformation</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="divider">------------------------</p>', unsafe_allow_html=True)
                st.sidebar.markdown(":red[Encoding categorical and numeric]")
                df = encoding_categorical(df)
                st.sidebar.markdown(":red[Scaling and normalizing]")
                df = scaler(df)
            st.header("")
            st.data_editor(df)
            st.header("")
            col2, col3 = st.columns([1, 1])
            with col2:
                display_data_information(df)
            with col3:
                display_missing_values(df)
            display_descriptive_statistics(df)

    with tab2:
        visualize_data(df)



    # Use the dot matrix font for your text


if __name__ == "__main__":
    main()
