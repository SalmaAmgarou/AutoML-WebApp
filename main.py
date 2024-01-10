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

def main():
    st.set_page_config(page_title="Machine Learning in Action", page_icon="💾")
    col1, col2 = st.columns([1, 1])  # Two columns of equal width
    with col1:
        st.image('images/LOGO IADS FR.png',  width=240)
    with col2:
        st.image('images/fstt.png',  width=420)
    tab1, tab2 = st.tabs(["Upload Data", "Visualize Data"])
    df = None
    with tab1:
        df = load_data()
        if df is not None:
            st.write(":red[Switch to the 'Visualize Data' tab to see the visualizations.]")
            columns = st.columns(2)

            with columns[0]:
                st.sidebar.header(":orange[Features selection]")
                st.sidebar.markdown(":red[Handle Missing Values (Numeric)]")
                df = handle_missing_values_numeric(df)
                st.sidebar.markdown(":red[Handle Missing Values (Categorical)]")
                df = handle_missing_values_categorical(df)
                st.sidebar.markdown(":red[Handle Outliers (Z-score) (Numeric)]")
                df = handle_outliers_zscore(df, threshold=3.0)
                st.sidebar.header(":orange[Data transformation]")
                st.sidebar.markdown(":red[Encoding categorical and numeric]")
                df = encoding_categorical(df)
                st.sidebar.markdown(":red[Scaling and normalizing]")
                df = scaler(df)
            st.header("")
            st.markdown("--------------------------------------------------UPDATED DATASET----------------------------------------------------------")
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

if __name__ == "__main__":
    main()
