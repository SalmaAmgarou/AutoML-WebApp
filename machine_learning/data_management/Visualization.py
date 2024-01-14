import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time


# Function to visualize data based on user-selected options

def visualize_data(dataframe):
    st.markdown('<p class="visualize">Visualization Options</p>', unsafe_allow_html=True)

    # Initialize selected options in session state if not present
    if 'selected_options' not in st.session_state:
        st.session_state.selected_options = []

    # Multiselect for choosing visualization options
    graphs_and_plots = st.multiselect(
        "Select Visualization Options",
        ["Correlation Heatmap", "Scatter Plot", "Histogram", "Area Chart", "Bar Plot"]
    )

    # Store selected options in session state
    st.session_state.selected_options = graphs_and_plots
    show_progress_bar()

    if "Bar Plot" in st.session_state.selected_options:
        st.header("")
        st.markdown("----------------------------------------------")
        plot_custom(dataframe)
        st.header("")
        st.markdown("----------------------------------------------")

    # Conditional rendering of visualizations based on user selection
    if "Correlation Heatmap" in st.session_state.selected_options:
        st.header("")
        st.markdown("----------------------------------------------")
        create_correlation_heatmap(dataframe)
        st.header("")
        st.markdown("----------------------------------------------")

    if "Scatter Plot" in st.session_state.selected_options:
        st.header("")
        st.markdown("----------------------------------------------")
        scatter_plot(dataframe)
        st.header("")
        st.markdown("----------------------------------------------")

    if "Area Chart" in st.session_state.selected_options:
        st.header("")
        st.markdown("----------------------------------------------")
        create_area_chart(dataframe)
        st.header("")
        st.markdown("----------------------------------------------")

    if "Histogram" in st.session_state.selected_options:
        st.header("")
        st.markdown("----------------------------------------------")
        seaborn_histogram(dataframe)
        st.header("")
        st.markdown("----------------------------------------------")


# Function to show a progress bar while generating visualizations
def show_progress_bar():
    progress_text = "Generating Visualization..."
    with st.spinner(progress_text):
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        time.sleep(0.01)
        my_bar.empty()



# Function to create a custom bar plot
def plot_custom(dataframe):
    st.subheader(":orange[Bar Plot]")
    st.markdown('<p class="visualize">Visualization Options</p>', unsafe_allow_html=True)
    st.header("")

    x = tuple(dataframe.columns)
    y = tuple(dataframe.columns)

    # Provide unique keys for each selectbox
    selected_x = st.selectbox('Select X for plot', x, key='select_x')
    selected_y = st.selectbox('Select Y for plot', y, key='select_y')

    if selected_y == selected_x:
        st.warning(f"Choose a different column for Y!")
    else:
        fig = px.bar(
            data_frame=dataframe, x=selected_x, y=selected_y
        )
        st.plotly_chart(fig)

# Function to create a correlation heatmap
def create_correlation_heatmap(dataframe):
    st.markdown('<p class="titles">Correlation Heatmap</p>', unsafe_allow_html=True)
    st.header("")

    if dataframe is None:
        st.write("No data available to generate a correlation heatmap.")
    else:
        numeric_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns

        try:
            corr_matrix = dataframe[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)

        except ValueError as e:
            st.warning(f"Warning: {str(e)}. Preprocessing of data is recommended before generating the correlation heatmap.")


# Function to create a scatter plot
def scatter_plot(dataframe):
    st.markdown('<p class="titles">Scatter Plot</p>', unsafe_allow_html=True)
    st.header("")

    numeric_cols = dataframe.select_dtypes(exclude="object").columns.to_list()
    var_x = st.selectbox("Select the Abscissa", numeric_cols)
    var_y = st.selectbox("Select the Oridinate", numeric_cols)
    categorical_cols = dataframe.select_dtypes(include="object").columns.to_list()
    var_col = st.selectbox("Variable for coloring the points", categorical_cols)

    fig2 = px.scatter(
        data_frame= dataframe,
        x=var_x,
        y=var_y,
        color=var_col,
        title= str(var_y) + " Vs " + str(var_x)
    )
    st.plotly_chart(fig2)


# Function to create an area chart
def create_area_chart(dataframe):
    st.markdown('<p class="titles">Area Chart</p>', unsafe_allow_html=True)
    st.header("")

    numeric_cols = dataframe.select_dtypes(exclude="object").columns.to_list()
    var_x = st.selectbox("Select the X-axis column", numeric_cols)
    var_y = st.multiselect("Select Y-axis columns", numeric_cols)

    if var_y:
        colors = ['#FF0000', '#0000FF']  # Default colors
        fig = px.area(dataframe, x=var_x, y=var_y, color_discrete_sequence=colors)
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least one Y-axis column")


# Function to create a Seaborn histogram
def seaborn_histogram(dataframe):
    st.markdown('<p class="titles">Seaborn Histogram</p>', unsafe_allow_html=True)
    st.header("")

    numeric_cols = dataframe.select_dtypes(include=['float64', 'int64'],exclude="object").columns

    selected_column = st.selectbox("Select a Column", numeric_cols)

    if selected_column in numeric_cols:
        fig_sb, ax_sb = plt.subplots()
        ax_sb = sns.histplot(dataframe[selected_column])
        plt.xlabel(selected_column)
        st.pyplot(fig_sb)
    else:
        st.warning("Selected column is not numeric. Please select a numeric column.")
