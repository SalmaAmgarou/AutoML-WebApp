import streamlit as st

def visualize_data(dataframe):
    st.subheader("Visualization Options")

    graphs_and_plots = st.multiselect("Select Visualization Options",
                                      ["Correlation Heatmap", "Scatter Plot", "Histogram", "Box Plot"])

    st.write("Visualization Options Selected:")
    st.write(graphs_and_plots)
