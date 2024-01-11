import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer



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

def plot_metrics(metrics_result):
    if metrics_result:
        st.subheader('Metrics')
        st.write(metrics_result)

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    st.subheader('Confusion Matrix')
    st.write(cm)

def plot_regression_results(y_true, y_pred, residuals):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Scatter plot of Predicted vs Actual values
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[0, 0])
    axes[0, 0].set_title('Predicted vs Actual')

    # Residual Plot
    sns.residplot(x=y_pred, y=residuals, ax=axes[0, 1])
    axes[0, 1].set_title('Residual Plot')

    # Distribution of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Residuals')

    # Regression Line Plot
    sns.lineplot(x=y_true, y=y_true, color='red', ax=axes[0, 0])

    # Actual vs Predicted with Error Bars (useful for uncertainty estimation)
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[1, 1], alpha=0.7)

    # Ensure that residuals are non-negative for error bars
    residuals_non_negative = np.abs(residuals)

    axes[1, 1].errorbar(x=y_true, y=y_pred, yerr=residuals_non_negative, fmt='o', markersize=0, color='red', alpha=0.5)
    axes[1, 1].set_title('Actual vs Predicted with Error Bars')

    plt.tight_layout()
    st.pyplot(fig)
def plot_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    st.subheader('Classification Report')
    st.text(report)

def plot_roc_curve(y_true, y_probas):
    st.subheader('ROC Curve')
    fig, ax = plt.subplots()
    skplt.metrics.plot_roc_curve(y_true, y_probas, ax=ax)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    st.pyplot(fig)


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
        metrics = ["MAE", "MSE", "RMSE", "R2 Square", "Cross Validation RMSE"]
    elif task == "Clustering":
        st.header('Select Clustering Model')
        model_option = "K-Means"
        metrics = ["inertia"]
        model = KMeans()

        # Allow user to input the range of clusters
        min_clusters = st.slider('Select the minimum number of clusters (k):', min_value=1, max_value=10, value=1)
        max_clusters = st.slider('Select the maximum number of clusters (k):', min_value=2, max_value=10, value=5)

        inertias = []

        for num_clusters in range(min_clusters, max_clusters + 1):
            model = KMeans(n_clusters=num_clusters)

            try:
                model.fit(X)
                inertia = model.inertia_
                inertias.append({"Number of Clusters (k)": num_clusters, "Inertia": inertia})
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")

        st.subheader("Inertia values for different numbers of clusters:")
        st.write(inertias)

        # Use the elbow method to find the optimal number of clusters
        visualizer = KElbowVisualizer(KMeans(), k=(min_clusters, max_clusters))
        visualizer.fit(X)
        visualizer.show()

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
        pass  # We already set up the K-Means model above
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
                print("Sizes - x_train:", len(X_train), "y_train:", len(y_train))
                # Add this line for debugging

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2_square = r2_score(y_test, y_pred)

                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())

                metrics_result = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Square": r2_square, "Cross Validation RMSE": cv_rmse}

                plot_metrics(metrics_result)
                plot_regression_results(y_test, y_pred, y_test - y_pred)


            elif task == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                metrics_result = {"Accuracy": accuracy}
                plot_confusion_matrix(y_test, y_pred)
                plot_classification_report(y_test, y_pred)
                plot_roc_curve(y_test, model.predict_proba(X_test))

            elif task == "Clustering":
                if model_option == "K-Means":
                    model.fit(X)
                    labels = model.predict(X)
                    inertia = model.inertia_
                    metrics_result = {"Inertia": inertia}
                    st.write(metrics_result)
                    # You can also display a plot of the clusters if you like
                    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
                    plt.xlabel('Feature 1')
                    plt.ylabel('Feature 2')
                    st.pyplot(plt)

            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"An error occurred during training: {str(e)}")
    else:
        st.warning("Please select a valid machine learning model.")
