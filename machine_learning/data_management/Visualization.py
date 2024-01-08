import streamlit as st

def visualize_data(dataframe):
    st.subheader("Visualization Options")

    if 'selected_options' not in st.session_state:
        st.session_state.selected_options = []

    graphs_and_plots = st.multiselect("Select Visualization Options",
                                      ["Correlation Heatmap", "Scatter Plot", "Histogram", "Box Plot"])

    st.write("Visualization Options Selected:")
    st.write(graphs_and_plots)

    st.session_state.selected_options = graphs_and_plots

