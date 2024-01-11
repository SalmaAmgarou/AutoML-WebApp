import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def select_target(dataframe):
    if dataframe is not None:
        st.header('Select column to predict')
        target = tuple(dataframe.columns)
        selected_target = st.selectbox('Select target for prediction', target)
        st.write('selected target:', selected_target)
        return selected_target

def split_data(dataframe):
    if dataframe is not None:
        st.header('Select train : test ratio')
        traintest = st.slider('train:test:', min_value=0, max_value=100, step=5, value=80)
        train_ratio = traintest / 100
        st.write('train ratio:', train_ratio)
        test_ratio = (100 - traintest) / 100
        st.write('test ratio:', test_ratio)
        return train_ratio

def select_model_and_train(df):
    st.markdown('<p class="titles">Data Modeling</p>', unsafe_allow_html=True)
    st.header("")

    if df is None:
        st.warning("Please upload and preprocess data first.")
        return

    selected_target = select_target(df)
    train_ratio = split_data(df)

    X = df.drop(columns=[selected_target])
    y = df[selected_target]

    train_ratio = float(train_ratio)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)

    model_option = st.radio("Select Model", ["Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression", "Support Vector Classifier", "K-Nearest Neighbors"])
    model = None

    if model_option == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif model_option == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "Support Vector Classifier":
        model = SVC()
    elif model_option == "K-Nearest Neighbors":
        model = KNeighborsClassifier()

    if model:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
        except Exception as e:
            st.error(f"An error occurred during training: {str(e)}")
    else:
        st.warning("Please select a valid machine learning model.")
