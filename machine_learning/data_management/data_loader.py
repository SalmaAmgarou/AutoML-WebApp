import streamlit as st
import pandas as pd
def load_data():

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None

    st.title("Machine Learning in Action")
    st.header("")
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

    if not uploaded_files:
        code = '''def hello():
        print("Hello, Please Upload a file!")'''
        st.code(code, language='python')
        return None

    st.session_state.uploaded_files = uploaded_files

    dataframes = []

    try:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension in ["xls", "xlsx"]:
                df = pd.read_excel(uploaded_file)
            elif file_extension == "json":
                df = pd.read_json(uploaded_file)
            else:
                st.warning(f"File format '{file_extension}' is not supported.")
                continue
            dataframes.append(df)

        if not dataframes:
            st.warning("No valid data found in the uploaded files.")
            return None  # Return None if no valid data is found
        st.write("")
        st.write("")
        st.subheader("① Edit and select cells")
        st.info("💡 You can edit the cells in the table below.")
        st.caption("")

        selected_df = pd.concat(dataframes)

        # Checkbox to enable column deletion
        delete_columns = st.checkbox("Enable Column Deletion")

        if delete_columns:
            # Multi-select box for choosing columns to delete
            columns_to_delete = st.multiselect("Select columns to delete", selected_df.columns.tolist())

            # Remove selected columns from the DataFrame
            selected_df = selected_df.drop(columns=columns_to_delete, errors='ignore')

        # Data editor for the remaining DataFrame
        edited_df = st.data_editor(selected_df, num_rows="dynamic")

        return edited_df

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
