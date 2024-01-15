import streamlit as st
import pandas as pd

def load_data():
    '''Explanation of the function's purpose, role, and usage in the application:

    Purpose: The purpose of this function is to load data from multiple file formats (CSV, Excel, JSON) using Streamlit's file uploader. It reads the uploaded files and concatenates them into a single DataFrame, which is then displayed in an editable data table.

    Role:
        Checks and initializes the 'uploaded_files' attribute in Streamlit session state.
        Allows users to upload multiple files.
        Handles different file formats (CSV, Excel, JSON).
        Concatenates DataFrames from uploaded files.
        Displays an editable data table for further user interaction.

    Usage in the Application: This function is used in the "Upload Data" tab of the Streamlit app. It enables users to upload, view, and edit datasets interactively. If no valid data is found, it displays a warning message. If an error occurs during the data loading process, it shows an error message.'''

    # Check if 'uploaded_files' is in session state; if not, initialize it
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None

    # Allow users to upload multiple files
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

    # If no files are uploaded, display a message and return None
    if not uploaded_files:
        code = '''def hello():
        print("Hello, Please Upload a file!")'''
        st.code(code, language='python')
        return None

    # Store uploaded files in session state
    st.session_state.uploaded_files = uploaded_files

    # List to store DataFrames from uploaded files
    dataframes = []

    try:
        # Iterate through uploaded files and read them based on file extensions
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()

            # Read CSV file
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            # Read Excel file
            elif file_extension in ["xls", "xlsx"]:
                df = pd.read_excel(uploaded_file)
            # Read JSON file
            elif file_extension == "json":
                df = pd.read_json(uploaded_file)
            else:
                # Warn if the file format is not supported
                st.warning(f"File format '{file_extension}' is not supported.")
                continue
            dataframes.append(df)

        # If no valid data is found, display a warning and return None
        if not dataframes:
            st.warning("No valid data found in the uploaded files.")
            return None  # Return None if no valid data is found


        # Concatenate DataFrames from all uploaded files
        selected_df = pd.concat(dataframes)

        # Display the data editor for further editing
        edited_df = st.data_editor(selected_df)

        return edited_df

    except Exception as e:
        # Display an error message if an exception occurs
        st.error(f"An error occurred: {str(e)}")
        return None