# Import necessary libraries
import streamlit as st
from user_interface.window_main import display_data_information, display_missing_values, display_descriptive_statistics
from machine_learning.data_management.data_loader import load_data
from machine_learning.data_management.data_modeling import select_model_and_train
from machine_learning.data_management.data_preprocessing import (
    delete_columns,
    handle_missing_values_numeric,
    handle_outliers,
    handle_missing_values_categorical,
    encoding_categorical,
    scaler,
)
import time
from machine_learning.data_management.Visualization import visualize_data
from Documentation import documentation
import streamlit as st
from streamlit_option_menu import option_menu as om




# Main function
def main():
    # Set page configuration and title
    st.set_page_config(page_title="Machine Learning in Action", page_icon="💾", menu_items={
        'About': "### Welcome to our Machine Learning in Action. \n *Version 1.0* - Created with passion by HAMZA HAFDAOUI [Hamza Hafdaoui on GitHub](https://github.com/HAMZAUEST) and SALMA AMGAROU.   [Salma Amgarou on GitHub](https://github.com/SalmaAmgarou) \n What is this? \n **Explore the fascinating world of machine learning through our Streamlit app! We've curated an interactive experience that allows you to witness machine learning algorithms in action."})
    # Styling with custom fonts
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
    col1, col2 = st.columns([1, 1])  # Two columns of equal width
    with col1:
        st.image('images/LOGO IADS FR.png',  width=240)
    with col2:
        st.image('images/fstt.png',  width=420)
    st.markdown('<p class="title">Machine Learning In Action</p>', unsafe_allow_html=True)
    # Option menu for selecting different sections
    selected = om("", ['', ' '], icons=['house', 'book', 'list-task', 'gear'], menu_icon='cast', default_index=0, orientation='horizontal')
    selected
    # Main section handling different tabs
    if selected == '':
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["Upload Data", "Visualize Data", "Data Modeling"])
        # Session state initialization
        if 'original_df' not in st.session_state:
            st.session_state.original_df = None
        if 'preprocessed_df' not in st.session_state:
            st.session_state.preprocessed_df = None
        if 'model' not in st.session_state:
            st.session_state.model = None  # Initialize model attribute

        # Tab 1: Upload Data
        with tab1:
            with st.status("Downloading data..."):
                # Simulate data loading delay
                time.sleep(0.8)
                st.session_state.original_df = load_data()

            # Check if data is loaded successfully
            if st.session_state.original_df is not None:
                st.success('Data Loaded Successfully!')
                st.write(":red[Switch to the 'Visualize Data' tab to see the visualizations.]")
                # Sidebar for data preprocessing options
                columns = st.columns(2)
                if st.session_state.preprocessed_df is None:
                    preprocessed_df = st.session_state.original_df.copy()
                else:
                    preprocessed_df = st.session_state.preprocessed_df.copy()

                st.sidebar.markdown('<p class="divider">####################</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="dot-matrix">Features selection</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="divider">####################</p>', unsafe_allow_html=True)

                # Enable column deletion checkbox
                delete_columns_checkbox = st.checkbox("Enable Column Deletion")
                if delete_columns_checkbox:
                    preprocessed_df = delete_columns(preprocessed_df)
                # Handle missing values (numeric) checkbox
                missing_values_numeric = st.sidebar.checkbox("Handle Missing Values (Numeric)")
                if missing_values_numeric:
                    preprocessed_df = handle_missing_values_numeric(preprocessed_df)

                # Handle missing values (categorical) checkbox
                missing_values_categorical = st.sidebar.checkbox("Handle Missing Values (Categorical)")
                if missing_values_categorical:
                    preprocessed_df = handle_missing_values_categorical(preprocessed_df)
                # Handle outliers checkbox
                handle_out = st.sidebar.checkbox("Handle Outliers")
                if handle_out:
                    preprocessed_df = handle_outliers(preprocessed_df, threshold=3.0)
                st.sidebar.markdown('<p class="divider">####################</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="dot-matrix">Data transformation</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="divider">####################</p>', unsafe_allow_html=True)

                # Encoding categorical and numeric checkbox
                encoding = st.sidebar.checkbox("Encoding categorical and numeric")
                if encoding:
                    preprocessed_df = encoding_categorical(preprocessed_df)
                scaling = st.sidebar.checkbox("Feature Scaling")
                if scaling:
                    preprocessed_df = scaler(preprocessed_df)

                # Feature scaling checkbox
                preprocessed_df = st.data_editor(preprocessed_df, num_rows="dynamic")

                # Display preprocessed data in data editor
                st.header("")
                col2, col3 = st.columns([1, 1])
                with col2:
                    display_data_information(preprocessed_df)
                with col3:
                    display_missing_values(preprocessed_df)
                display_descriptive_statistics(preprocessed_df)
                # Save preprocessed data in session state
                st.session_state.preprocessed_df = preprocessed_df
        # Tab 2: Visualize Data
        with tab2:
            visualize_data(st.session_state.preprocessed_df)

        # Tab 3: Data Modeling
        with tab3:
            st.header("")
            st.markdown('<p class="titles">Data Modeling</p>', unsafe_allow_html=True)
            st.header("")

            # Check if preprocessed data is available
            if st.session_state.preprocessed_df is None:
                st.warning("Please upload and preprocess data first.")
            else:
                # Radio button to select the task (Classification, Regression, Clustering)
                task = st.radio("Select Task", ["Classification", "Regression", "Clustering"])

                # Select model and train
                select_model_and_train(st.session_state.preprocessed_df, task)

    # Documentation section
    elif selected == ' ':
        documentation()



# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()

