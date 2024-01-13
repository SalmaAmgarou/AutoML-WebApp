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
import os
import streamlit as st
from streamlit_option_menu import option_menu as om





def main():

    st.set_page_config(page_title="Machine Learning in Action", page_icon="💾", menu_items={
        'About': "### Welcome to our Machine Learning in Action. \n *Version 1.0* - Created with passion by HAMZA HAFDAOUI [Hamza Hafdaoui on GitHub](https://github.com/HAMZAUEST) and SALMA AMGAROU.   [Salma Amgarou on GitHub](https://github.com/SalmaAmgarou) \n What is this? \n **Explore the fascinating world of machine learning through our Streamlit app! We've curated an interactive experience that allows you to witness machine learning algorithms in action."})
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
    selected = om("", ['', ' '], icons=['house', 'book', 'list-task', 'gear'], menu_icon='cast', default_index=0, orientation='horizontal')
    selected
    if selected == '':



        tab1, tab2, tab3 = st.tabs(["Upload Data", "Visualize Data", "Data Modeling"])

        if 'original_df' not in st.session_state:
            st.session_state.original_df = None
        if 'preprocessed_df' not in st.session_state:
            st.session_state.preprocessed_df = None
        if 'model' not in st.session_state:
            st.session_state.model = None  # Initialize model attribute

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
                st.sidebar.markdown('<p class="divider">####################</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="dot-matrix">Features selection</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="divider">####################</p>', unsafe_allow_html=True)
                delete_columns_checkbox = st.checkbox("Enable Column Deletion")
                if delete_columns_checkbox:
                    preprocessed_df = delete_columns(preprocessed_df)
                missing_values_numeric = st.sidebar.checkbox("Handle Missing Values (Numeric)")
                if missing_values_numeric:
                    preprocessed_df = handle_missing_values_numeric(preprocessed_df)
                missing_values_categorical = st.sidebar.checkbox("Handle Missing Values (Categorical)")
                if missing_values_categorical:
                    preprocessed_df = handle_missing_values_categorical(preprocessed_df)
                handle_out = st.sidebar.checkbox("Handle Outliers")
                if handle_out:
                    preprocessed_df = handle_outliers(preprocessed_df, threshold=3.0)
                st.sidebar.markdown('<p class="divider">####################</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="dot-matrix">Data transformation</p>', unsafe_allow_html=True)
                st.sidebar.markdown('<p class="divider">####################</p>', unsafe_allow_html=True)
                encoding = st.sidebar.checkbox("Encoding categorical and numeric")
                if encoding:
                    preprocessed_df = encoding_categorical(preprocessed_df)
                scaling = st.sidebar.checkbox("Feature Scaling")
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
        with tab3:
            st.header("")
            st.markdown('<p class="titles">Data Modeling</p>', unsafe_allow_html=True)
            st.header("")

            if st.session_state.preprocessed_df is None:
                st.warning("Please upload and preprocess data first.")
            else:
                task = st.radio("Select Task", ["Classification", "Regression", "Clustering"])
                select_model_and_train(st.session_state.preprocessed_df, task)
    elif selected == ' ':
        st.title("Documentation and Tutorials")

        st.header("Code Comments and Docstrings")
        st.markdown("""
        - Ensure that the code is well-commented. Use comments to explain complex or non-trivial sections of the code.
        - Utilize docstrings for functions and classes, providing information on parameters, return values, and the purpose of the function or class.
        """)

        st.header("Separate Documentation Section")
        st.markdown("""
        - Create a dedicated section for documentation and tutorials accessible through a navigation menu or a separate tab.
        - Use Markdown for clear and organized documentation.
        """)

        st.header("Tutorials")
        st.markdown("""
        - Provide step-by-step tutorials for common tasks and workflows.
        - Include a combination of text, code snippets, and visuals to guide users through the process.
        """)

        st.header("Algorithm Explanations")
        st.markdown("""
        - Include detailed explanations for each implemented algorithm.
        - Describe the purpose, functionality, and relevant parameters of each algorithm.
        - Provide references to external resources or papers for further reading.
        """)

        st.header("Interactive Examples")
        st.markdown("""
        - Use Streamlit's interactive features to create examples that users can manipulate in real-time.
        - Enhance tutorials with interactive elements to improve user engagement.
        """)

        st.header("Sample Datasets")
        st.markdown("""
        - Include sample datasets that users can use to follow along with tutorials and test the application's functionalities.
        - Provide information on the structure and content of sample datasets.
        """)

        st.header("User Feedback")
        st.markdown("""
        - Allow users to provide feedback on documentation and tutorials.
        - Implement a feedback mechanism to gather user opinions and suggestions.
        """)

        st.header("Version Control")
        st.markdown("""
        - Ensure that documentation is updated with each version of the application.
        - Users should be able to access documentation corresponding to the version they are using.
        """)

        st.header("Additional Resources")
        st.markdown("""
        - Provide links to external resources or documentation for related topics, such as machine learning concepts, Streamlit usage, or data science best practices.
        """)

        st.header("Community Support")
        st.markdown("""
        - Consider creating a community forum or space for users to ask questions, share tips, and discuss the application.
        - Foster a sense of community and provide additional support through user interactions.
        """)

        st.success("Documentation and tutorials have been outlined. Customize them based on your application's specifics.")


if __name__ == "__main__":
    main()

