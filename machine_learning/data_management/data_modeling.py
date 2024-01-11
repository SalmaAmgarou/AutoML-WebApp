import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np

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

def select_model_and_train(df, task):
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

    model_option = None
    metrics = None

    if task == "Classification":
        st.header('Select Classification Model')
        model_option = st.selectbox("Select Model", ["Decision Trees", "Naive Bayes", "Support Vector Machine (SVM)", "K-Nearest Neighbors", "Random Forest", "Artificial Neural Networks"])
        metrics = ["accuracy"]
    elif task == "Regression":
        st.header('Select Regression Model')
        model_option = st.selectbox("Select Model", ["Linear Regression", "Decision Trees", "Support Vector Machine (SVM)", "K-Nearest Neighbors", "Random Forest", "Artificial Neural Networks"])
        metrics = ["MAE", "RMSE"]
    elif task == "Clustering":
        st.header('Select Clustering Model')
        model_option = "K-Means"
        metrics = ["inertia"]

    model = None
    metrics_result = None

    if model_option == "Linear Regression":
        model = LinearRegression()
    elif model_option == "Decision Trees":
        if task == "Regression":
            model = DecisionTreeRegressor()
        elif task == "Classification":
            model = DecisionTreeClassifier()
    elif model_option == "Naive Bayes":
        model = GaussianNB()
    elif model_option == "Support Vector Machine (SVM)":
        if task == "Regression":
            model = SVR()
        elif task == "Classification":
            model = SVC()
    elif model_option == "K-Means":
        if task == "Clustering":
            model = KMeans()
    elif model_option == "K-Nearest Neighbors":
        if task == "Regression":
            model = KNeighborsRegressor()
        elif task == "Classification":
            model = KNeighborsClassifier()
    elif model_option == "Random Forest":
        if task == "Regression":
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor()
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            st.write("Best parameters:", grid_search.best_params_)
        elif task == "Classification":
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier()
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            st.write("Best parameters:", grid_search.best_params_)
    elif model_option == "Artificial Neural Networks":
        if task == "Regression":
            model = MLPRegressor(max_iter=500)
        elif task == "Classification":
            model = MLPClassifier(max_iter=500)

    if model:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task == "Regression":
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics_result = {"MAE": mae, "RMSE": rmse}
            elif task == "Classification":
                accuracy = (y_test == y_pred).mean()
                metrics_result = {"Accuracy": accuracy}
            elif task == "Clustering":
                if model_option == "K-Means":
                    inertia = model.inertia_
                    metrics_result = {"Inertia": inertia}

            st.success(f"Model trained successfully! Metrics: {metrics_result}")
        except Exception as e:
            st.error(f"An error occurred during training: {str(e)}")
    else:
        st.warning("Please select a valid machine learning model.")
