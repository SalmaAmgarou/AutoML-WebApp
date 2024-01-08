import streamlit as st
import matplotlib.pyplot as plt
from machine_learning.data_management.data_loader import load_data
from machine_learning.data_management.data_preprocessing import select_target
from machine_learning.data_management.Visualization import visualize_data , create_correlation_heatmap

if __name__ == "__main__":
    st.set_page_config(page_title="Machine Learning in Action", page_icon="💾")
    tab1, tab2 = st.tabs(["Upload Data", "Visualize Data"])

    df = None

    with tab1:
        df = load_data()
        if df is not None:
            st.write("Switch to the 'Visualize Data' tab to see the visualizations.")
            select_target(df)

    with tab2:
        visualize_data(df)

