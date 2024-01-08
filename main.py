import streamlit as st
import matplotlib.pyplot as plt
from user_interface.window_main import display_data_information , display_missing_values , display_descriptive_statistics
from machine_learning.data_management.data_loader import load_data
from machine_learning.data_management.data_preprocessing import select_target
from machine_learning.data_management.Visualization import visualize_data , create_correlation_heatmap
from images import  *

if __name__ == "__main__":
    st.set_page_config(page_title="Machine Learning in Action", page_icon="💾")
    col1, col2 = st.columns(2)
with col1:
    st.image('/home/hafdaoui/PycharmProjects/RAFINHA/images/LOGO IADS FR.png',caption=None, width=30)

# Right-aligned image
with col2:
    st.image('/home/hafdaoui/PycharmProjects/RAFINHA/images/fstt.png',caption=None, width=30)
    tab1, tab2 = st.tabs(["Upload Data", "Visualize Data"])
    df = None
    with tab1:
        df = load_data()
        if df is not None:
            st.write(":red[Switch to the 'Visualize Data' tab to see the visualizations.]")
            columns = st.columns(2)  # Use st.columns instead of st.beta_columns

            with columns[0]:
                display_data_information(df)


            with columns[1]:
                display_missing_values(df)


            display_descriptive_statistics(df)
            select_target(df)



    with tab2:
        visualize_data(df)

