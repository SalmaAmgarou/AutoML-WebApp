import streamlit as st

# Application Documentation
def documentation():
    st.markdown('<p class="title">Application Documentation</p>', unsafe_allow_html=True)
    st.write('Table of Contents')

    # Introduction
    st.markdown("""
        1. [Introduction](#introduction)
        2. [Code Comments](#code-comments)
        3. [Data Modeling](#data-modeling)
        4. [Algorithm Explanations](#algorithm-explanations)
        5. [Usability Testing](#usability-testing)
        6. [User Guide](#user-guide)
        7. [Feedback and Support](#feedback-and-support)
    """)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Introduction", "Tutorial", "Data Modeling", "Algorithm Explanations", "Usability Testing", "User Guide", "Feedback and Support"])

    # Introduction
    with tab1:
        st.markdown('<p class="title">Introduction</p>', unsafe_allow_html=True)
        st.markdown("""
        Welcome to our application documentation! This guide is designed to help you understand and make the most out of our interactive data exploration and machine learning tool.
        """)

        # Key Features
        st.markdown('<p class="titles">Key Features</p>', unsafe_allow_html=True)
        st.markdown("""
        - **Data Preprocessing:** Easily delete columns, handle missing values, and scale numerical features with just a few clicks.
        - **Data Modeling:** Train machine learning models for regression, classification, or clustering tasks using a variety of algorithms.
        - **Visualization:** Create insightful visualizations, including scatter plots, correlation heatmaps, and more, to gain a deeper understanding of your data.
        """)

        # Prerequisites
        st.markdown('<p class="titles">Prerequisites</p>', unsafe_allow_html=True)
        st.markdown("""
        Before you start using the application, make sure you have:
        - A compatible web browser (we recommend the latest versions of Chrome or Firefox).
        - A dataset in a supported format (CSV, Excel, etc.) for data preprocessing and modeling.
        """)

        # Getting Started
        st.markdown('<p class="titles">Getting Started</p>', unsafe_allow_html=True)
        st.markdown("""
        To begin your data exploration journey, follow our step-by-step guide in the [User Guide](#user-guide). If you encounter any issues, refer to the [Feedback and Support](#feedback-and-support) section to get in touch with us.
    
        Happy exploring!
        """)

    # Code Comments
    with tab2:
        st.markdown('<p class="title">Tutorials</p>', unsafe_allow_html=True)
        st.write("""
        Tutorials are a great way to quickly learn how to use our application. We have provided video tutorials to guide you through various features and functionalities. Watch the tutorials to make the most out of the application.
        """)

        st.markdown('<p class="titles">Video Tutorial: Getting Started</p>', unsafe_allow_html=True)
        st.write("""
        [![Getting Started](insert_youtube_thumbnail_url)](insert_youtube_video_url)
        
        In this tutorial, we walk you through the process of getting started with our application. Learn how to upload your dataset, visualization techniques, and explore the key features.
        """)

        st.markdown('<p class="titles">Video Tutorial: Data Modeling</p>', unsafe_allow_html=True)
        st.write("""
        [![Data Modeling](insert_youtube_thumbnail_url)](insert_youtube_video_url)
        
        Explore the data modeling capabilities of our application. Understand how to select target columns, choose machine learning models, and evaluate their performance.
        """)

        st.markdown('<p class="titles">Video Tutorial: Advanced Features</p>', unsafe_allow_html=True)
        st.write("""
        [![Advanced Features](insert_youtube_thumbnail_url)](insert_youtube_video_url)
        
        Dive into advanced features of the application. This tutorial covers advanced data preprocessing, visualization techniques, and tips for optimizing your machine learning workflow.
        """)

        st.markdown('<p class="titles">Getting Help</p>', unsafe_allow_html=True)
        st.write("""
        If you encounter any challenges or have specific questions, refer to the [Feedback and Support](#feedback-and-support) section for assistance. Additionally, feel free to explore our comprehensive [Documentation](#documentation) for detailed guides.
        """)

        st.write("""
        Happy learning and exploring!
        """)

    # Data Modeling
    with tab3:
        st.markdown('<p class="title">Data Modeling</p>', unsafe_allow_html=True)
        st.write("""
        In the "Data Modeling" tab, you can train machine learning models for classification, regression, or clustering. Key steps include:
        1. Select the target column for prediction.
        2. Adjust the train-test split ratio.
        3. Choose a machine learning model based on your task (classification, regression, or clustering).
        4. View model metrics and visualizations.
        """)

        st.markdown('<p class="titles">Selecting Target Column</p>', unsafe_allow_html=True)
        st.write("""
        Before starting the modeling process, choose the target column you want to predict. This is the variable the model will learn to predict based on other features in the dataset.
        """)

        st.markdown('<p class="titles">Train-Test Split</p>', unsafe_allow_html=True)
        st.write("""
        Adjust the ratio for splitting your dataset into training and testing sets. This step is crucial for evaluating the model's performance on unseen data.
        """)

        st.markdown('<p class="titles">Choosing a Model</p>', unsafe_allow_html=True)
        st.write("""
        Select a machine learning model based on your task. The application supports various algorithms for classification, regression, and clustering. Experiment with different models to find the one that best suits your data.
        """)

        st.markdown('<p class="titles">Viewing Metrics and Visualizations</p>', unsafe_allow_html=True)
        st.write("""
        After training a model, explore the model metrics and visualizations to assess its performance. Common metrics include accuracy, precision, recall, and F1 score for classification tasks. Regression tasks may include metrics like mean squared error and R-squared.
        """)

        st.write("""
        The visualizations provided can include confusion matrices, ROC curves, and precision-recall curves. Understanding these metrics and visualizations will help you make informed decisions about the model's effectiveness.
        """)

        st.write("""
        For more detailed information on each algorithm's parameters, consult the [Algorithm Explanations](#algorithm-explanations) section.
        """)

    # Algorithm Explanations
    with tab4:
        st.markdown('<p class="title">Algorithm Explanations</p>', unsafe_allow_html=True)
        st.write("""
        In this section, we provide detailed explanations for the machine learning algorithms implemented in our application. Understanding the underlying algorithms can help users make informed choices when selecting models for their specific tasks.
        """)

        st.markdown('<p class="titles">Linear Regression</p>', unsafe_allow_html=True)
        st.write("""
        Linear regression predicts a Y value, given X features. Machine learning works to show the relationship between the two, then the relationships are placed on an X/Y axis, with a straight line running through them to predict future relationships.
In sentiment analysis, linear regression calculates how the X input (meaning words and phrases) relates to the Y output (opinion polarity – positive, negative, neutral). This will determine where the text falls on the scale of “very positive” to “very negative” and between.
        """)
        st.header("")
        st.image('images/regression_lin.png', caption='Linear Regression', use_column_width=True)
        st.header("")
        st.caption("Example of Linear Regression in Practice")
        code = '''import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Generate some random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Train a linear regression model model = LinearRegression() model.fit(x, y)
# Make predictions
X_new = np.array([[0], [2]])
y_pred
model.predict(X_new)
# Visualize the data and the linear regression line plt.scatter(x, y, label='Data points')
plt.plot(X_new, y_pred, 'r-',
label='Linear Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Example')
plt.show()'''
        st.code(code, language='python')


        st.markdown('<p class="titles">Logistique Regression</p>', unsafe_allow_html=True)
        st.write("""
        Logistic regression is an algorithm that predicts binary outcome, a positive or negative conclusion: Yes/No, Existence/Non-existence, Pass/Fail. It means, simply, something happens or does not.
Variables are calculated against each other to determine the 0/1 outcome (one of two categories):
P(Y=1|X) or P(Y=0|X)
The independent variables can be categorical or numeric, but the dependent variable is always categorical: the probability of dependent variable Y, given independent variable X. 
This can be used to determine the object in a photo or video image (cup, bowl, spoon, etc.) with each object given a probability between 0 and 1, or to calculate the probability of a word having a positive or negative connotation (0, 1, or on a scale between). """)
        st.header("")
        st.image('images/logistic-regression-in-machine-learning.png', caption='Logistique Regression', use_column_width=True)
        st.header("")
        st.caption("Example of Logistique Regression in Practice")
        code = '''import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate some synthetic data for binary classification
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
X_new = np.linspace(-3, 3, 300).reshape(-1, 1)
y_proba = model.predict_proba(X_new)

# Visualize the data and the logistic regression curve
plt.scatter(X, y, label='Data points', marker='o')
plt.plot(X_new, y_proba[:, 1], 'g-', label='Logistic Regression Curve (Class 1 Probability)')
plt.plot(X_new, y_proba[:, 0], 'b--', label='Class 0 Probability')
plt.xlabel('Feature (X)')
plt.ylabel('Probability')
plt.legend()
plt.title('Logistic Regression Example')
plt.show()'''
        st.code(code, language='python')

        st.markdown('<p class="titles">Decision Trees</p>', unsafe_allow_html=True)
        st.write("""
        Decision tree, also known as classification and regression tree (CART), is a supervised learning algorithm that works great on text classification problems because it can show similarities and differences on a hyper minute level. It, essentially, acts like a flow chart, breaking data points into two categories at a time, from “trunk,” to “branches,” then “leaves,” where the data within each category is at its most similar.

This creates classifications within classifications, showing how the precise leaf categories are ultimately within a trunk and branch category.
        """)
        st.header("")
        st.image('images/Decision_tree.png', caption='Decision Tree', use_column_width=True)
        st.header("")
        st.caption("Example of Decision Tree in Practice")
        code = '''# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn import tree
import graphviz
# Load a sample dataset (for example, the Iris dataset) iris datasets. load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
# Create a Decision Tree model dt_model = DecisionTreeClassifier() dt_model.fit(X_train, y_train)
# Visualize the Decision Tree as text
tree_rules = export_text(dt_model, feature_names-iris.feature_names) 
print("Decision Tree Rules: ", tree_rules)
# Visualize the Decision Tree as a graph
dot_data = tree.export_graphviz (dt_model, out_file=None,nfeature_names-iris.feature_names,
class_names-iris.target_names,nfilled=True,nrounded=True,nspecial_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree") # This will save a file named "iris_decision_tree.pdf"
# Display the Decision Tree graph (requires Graphviz) 
graph.view("iris_decision_tree")'''
        st.code(code, language='python')

        st.markdown('<p class="titles">Naive Bayes</p>', unsafe_allow_html=True)
        st.write("""
        When used in text analysis, Naive Bayes is a family of probabilistic algorithms that use Bayes’ Theorem to calculate the possibility of words or phrases falling into a set of predetermined “tags” (categories) or not. This can be used on news articles, customer reviews, emails, general documents, etc.

They calculate the probability of each tag for a given text, then output for the highest probability:
        """)
        st.image('images/relation_naive_bayes.png',use_column_width=True)
        st.write("""
        There can be multiple tags assigned to any given use case (problem), but they are each calculated individually. When tagging customer reviews, for example, we could use tags like Pricing, Usability, Features, etc., but each piece of text would be calculated against only one tag at a time:
        """)
        st.image('images/Naive_bapng.png', caption='Naive Bayes',use_column_width=True)
        st.header("")
        st.caption("Example of Naive Bayes in Practice")
        code = '''# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Visualize the data (2D plot for simplicity)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolor='k', s=100)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title(f'Naive Bayes Classification\nAccuracy: {accuracy:.2f}')
plt.show()'''
        st.code(code, language='python')

        st.markdown('<p class="titles">Support Vector Machine (SVM)</p>', unsafe_allow_html=True)
        st.write("""
        SVM is a powerful algorithm for both classification and regression tasks. It finds the hyperplane that best separates classes in the feature space. SVM can handle non-linear relationships through the use of kernel functions.
        """)
        st.header('')
        st.image('images/support-vector-machine-algorithm.png', caption='Support Vector Machine', use_column_width=True)
        st.header("")
        st.caption("Example of Support Vector Machine (SVM) in Practice")
        code = '''from sklearn import svm import numpy as np
import matplotlib.pyplot as plt
# Create a simple linearly separable dataset
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]]) 
y = np.array([1, 1, 1, 0, 0])
# Create a linear SVM model 
model = svm.SVC (kernel='linear')
# Train the model 
model.fit(x, y)
# Plot the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
ax = plt.gca() 
xlim = ax.get_xlim() 
ylim = ax.get_ylim()
# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = model.decision_function (np.c_[xx.ravel(), yy.ravel()])
# Plot decision boundary and margins
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('Linear SVM Example')
plt.show()'''
        st.code(code, language='python')


        st.markdown('<p class="titles">K-Means Clustering</p>', unsafe_allow_html=True)
        st.write("""
        K-Means is an unsupervised clustering algorithm that partitions data into k clusters. It minimizes the sum of squared distances between data points and their assigned cluster centroids. K-Means is widely used for data segmentation.
        """)
        st.header('')
        st.image('images/Kmeans.png', caption='K-Means Clustering', use_column_width=True)
        st.header("")
        st.caption("Example of Support Vector Machine (SVM) in Practice")
        code = '''import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create a custom dataset
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 0, 0])

# Train a KMeans model
model = KMeans(n_clusters=2, random_state=42)
model.fit(X)

# Get cluster labels
labels = model.labels_

# Visualize the data and clustering results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clustering Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
    '''
        st.code(code, language='python')

        st.markdown('<p class="titles">K-Nearest Neighbors (KNN)</p>', unsafe_allow_html=True)
        st.write("""
        K-nearest neighbors or “k-NN” is a pattern recognition algorithm that uses training datasets to find the k closest related members in future examples. 

Used in text analysis, we would calculate to place a given word or phrase within the category of its nearest neighbor. K is decided by a plurality vote of its neighbors.  If k = 1, then it would be placed in the class nearest 1.
        """)
        st.header('')
        st.image('images/knn-1.png', caption='K-Nearest Neighbors (KNN)', use_column_width=True)
        st.header("")
        st.caption("Example of K-Nearest Neighbors (KNN) in Practice")
        code = '''import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Create a custom dataset
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 0, 0])

# Train a KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Create a meshgrid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualize the data points and decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
plt.title('K-Nearest Neighbors (KNN) Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
    '''
        st.code(code, language='python')

        st.markdown('<p class="titles">Random Forest</p>', unsafe_allow_html=True)
        st.write("""
        Random forest is an expansion of decision tree and useful because it fixes the decision tree’s dilemma of unnecessarily forcing data points into a somewhat improper category.

It works by first constructing decision trees with training data, then fitting new data within one of the trees as a “random forest.” Put simply, random forest averages your data to connect it to the nearest tree on the data scale.""")
        st.header('')
        st.image('images/random-forest.jpg', caption='Random Forest', use_column_width=True)
        st.header("")
        st.caption("Example of Random Forest in Practice")
        code = '''import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Create a custom dataset
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 0, 0])

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a meshgrid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualize the data points and decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
plt.title('Random Forest Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
    '''
        st.code(code, language='python')

        st.markdown('<p class="titles">Artificial Neural Networks (ANN)</p>', unsafe_allow_html=True)
        st.write("""
        ANN is a deep learning algorithm inspired by the human brain's neural structure. It consists of interconnected layers of neurons, each performing weighted computations. ANNs are powerful for complex tasks but require careful tuning.
        """)
        st.header("")
        st.image('images/The-layout-of-the-deep-backpropagating-artificial-neural-network-ANN-using-TensorFlow.png', caption='Artificial Neural Networks (ANN)', use_column_width=True)
        st.caption("Example of Artificial Neural Networks (ANN) in Practice")
        code = '''import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Create a custom dataset
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 0, 0])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an Artificial Neural Network model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=0)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Create a meshgrid for plotting decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on the meshgrid
Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualize the data points and decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
plt.title('Artificial Neural Network (ANN) Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

    '''
        st.code(code, language='python')

        st.write("""
        :red[Understanding these algorithms will enable you to make informed decisions when selecting models in the application. For more detailed information on each algorithm's parameters, consult the [Model Training](#model-training) section.]
        """)

    # Usability Testing
    with tab5:
        st.markdown('<p class="title">Usability Testing</p>', unsafe_allow_html=True)
        st.write("""
        Usability testing is a critical phase in ensuring that our application meets the needs of users and provides a seamless and intuitive experience. This section outlines the usability testing process and the key findings that shaped the application's user interface and functionality.
        """)

        st.markdown('<p class="titles">Testing Methodology</p>', unsafe_allow_html=True)
        st.write("""
        Usability testing involves evaluating the application's user interface, navigation, and overall user experience. Our testing methodology includes the following steps:
        """)

        st.markdown("1. **Define User Scenarios:** We created realistic scenarios that users might encounter while using the application. These scenarios cover common tasks and functionalities.")
        st.markdown("2. **Recruit Test Participants:** We recruited a diverse group of participants representing the target user base. Participants included individuals with varying levels of technical expertise and domain knowledge.")
        st.markdown("3. **Conduct Testing Sessions:** Participants were asked to perform specific tasks using the application while providing feedback on their experiences. Observers noted areas of confusion, success, and suggestions for improvement.")
        st.markdown("4. **Collect Feedback:** Both quantitative and qualitative feedback was collected during testing sessions. Metrics such as task completion time, error rates, and user satisfaction were recorded.")
        st.markdown("5. **Iterative Design:** Based on the feedback received, we made iterative design changes to improve the user interface, enhance navigation, and address any usability issues.")

        st.markdown('<p class="titles">Key Findings</p>', unsafe_allow_html=True)
        st.write("""
        The usability testing process revealed valuable insights into how users interact with our application. Here are some key findings:
        """)

        st.markdown("1. **Intuitive Navigation:** Users found the navigation within the application to be intuitive, with clear pathways to access different features and functionalities.")
        st.markdown("2. **User-Friendly Interface:** The interface was well-received for its simplicity and clarity. Participants appreciated the clean design and easy-to-understand visualizations.")
        st.markdown("3. **Task Efficiency:** Users were able to complete tasks efficiently, with minimal confusion. The application's responsiveness contributed to a positive user experience.")
        st.markdown("4. **User Guidance:** Participants expressed the need for more in-app guidance, especially for users less familiar with machine learning concepts. We addressed this by enhancing tooltips and providing explanatory text where necessary.")

        st.markdown('<p class="titles">User Feedback</p>', unsafe_allow_html=True)
        st.write("""
        We value user feedback and encourage users to share their thoughts, suggestions, and any issues encountered while using the application. Continuous feedback helps us refine and improve the application over time.
        """)

        st.markdown('<p class="titles">Future Usability Testing</p>', unsafe_allow_html=True)
        st.write("""
        As we introduce new features and updates, we will continue to conduct usability testing to ensure that the application remains user-friendly and aligned with user expectations.
        """)

    # User Guide
    with tab6:
        st.markdown('<p class="title">User Guide</p>', unsafe_allow_html=True)
        st.write("""
        Welcome to the user guide for our application! This guide is designed to help you navigate the application, understand its features, and make the most out of the functionalities provided.
        """)

        st.markdown('<p class="titles">Table of Contents</p>', unsafe_allow_html=True)
        st.markdown("""
        1. [Getting Started](#getting-started)
        2. [Uploading Data](#uploading-data)
        3. [Data Preprocessing](#data-preprocessing)
        4. [Visualize Data](#visualize-data)
        5. [Data Modeling](#data-modeling)
        6. [Algorithm Explanations](#algorithm-explanations)
        7. [Usability Testing](#usability-testing)
        """)

        st.markdown('<p class="titles">Getting Started</p>', unsafe_allow_html=True)

        st.markdown('<p class="titles">Prerequisites</p>', unsafe_allow_html=True)
        st.write("""
        Before you begin, make sure you have the following:
        - A modern web browser (Chrome, Firefox, or Safari recommended)
        - An understanding of basic machine learning concepts
        """)

        st.markdown('<p class="titles">Accessing the Application</p>', unsafe_allow_html=True)
        st.markdown("""
        1. Visit the application's URL (insert URL here).
        2. You will be greeted with the home page, providing an overview of the application's capabilities.
        """)

        st.markdown('<p class="titles">Uploading Data</p>', unsafe_allow_html=True)

        st.write("To use the application effectively, you need to upload your dataset. Follow these steps:")
        st.markdown("1. Click on the \"Upload Data\" tab in the navigation menu.")
        st.markdown("2. Choose a file using the \"Choose File\" button.")
        st.markdown("3. Select the appropriate options for data preprocessing.")
        st.markdown("4. Click the \"Preprocess Data\" button to proceed.")

        st.markdown('<p class="titles">Data Preprocessing</p>', unsafe_allow_html=True)
        st.write("""
        The "Data Preprocessing" tab allows you to clean and prepare your dataset for modeling. Key functionalities include:
        - **Delete Columns:** Remove unwanted columns from your dataset.
        - **Handle Missing Values:** Choose techniques for handling missing values in both numeric and categorical columns.
        - **Handle Outliers:** Detect and handle outliers in numeric columns.
        - **Encoding Categorical:** Encode categorical columns using label or one-hot encoding.
        - **Scaler:** Scale numerical columns for improved model performance.
        """)

        st.markdown('<p class="titles">Visualize Data</p>', unsafe_allow_html=True)
        st.write("""
        Explore and visualize your dataset in the "Visualize Data" tab. Options include:
        - **Correlation Heatmap:** Understand relationships between numeric features.
        - **Scatter Plot:** Visualize the distribution and relationships between numeric features.
        - **Area Chart:** Create area charts for selected columns.
        - **Seaborn Histogram:** Generate histograms using the Seaborn library.
        - **Bar Plot:** Customize bar plots for specified columns.
        """)

        st.markdown('<p class="titles">Data Modeling</p>', unsafe_allow_html=True)
        st.write("""
        In the "Data Modeling" tab, you can train machine learning models for classification, regression, or clustering. Key steps include:
        1. Select the target column for prediction.
        2. Adjust the train-test split ratio.
        3. Choose a machine learning model based on your task (classification, regression, or clustering).
        4. View model metrics and visualizations.
        """)

        st.markdown('<p class="titles">Algorithm Explanations</p>', unsafe_allow_html=True)
        st.write("""
        This section provides detailed explanations of the algorithms implemented in the application. Understand the underlying principles and considerations for each algorithm.
        """)

        st.markdown('<p class="titles">Usability Testing</p>', unsafe_allow_html=True)
        st.write("""
        Explore the results of usability testing, including insights from user feedback, key findings, and improvements made based on user experiences.
        """)

        st.write("""
        Feel free to provide feedback, report issues, or suggest enhancements through the application. We value your input and are committed to delivering an excellent user experience.
        """)


    # Feedback and Support
    with tab7:
        st.markdown('<p class="title">Feedback and Support</p>', unsafe_allow_html=True)
        st.write("""
        We value your feedback and are here to provide support as you use our application. Your insights, questions, and suggestions help us improve and enhance the user experience. Feel free to reach out to us through the following channels:
        """)

        st.markdown('<p class="titles">Providing Feedback</p>', unsafe_allow_html=True)
        st.write("""
        We welcome any feedback you have regarding the application's functionality, usability, or any issues you may encounter. Your input is essential in helping us identify areas for improvement and deliver a better user experience.
        """)

        st.write("""
        To provide feedback:
        1. Click on the "Feedback" tab in the application.
        2. Fill out the feedback form, including details about your experience and any specific issues you encountered.
        3. Click the "Submit Feedback" button.
        """)

        st.markdown('<p class="titles">Support</p>', unsafe_allow_html=True)
        st.write("""
        If you need assistance or have questions about using the application, we're here to help! Reach out to us through one of the following support channels:
        """)

        st.markdown('<p class="titles">Email Support</p>', unsafe_allow_html=True)
        st.write("""
        - **Email Support:** Contact our support team via email at [hamza.hafdaoui@etu.uae.ac.ma](mailto:support@example.com) or [salma.amgarou@etu.uae.ac.ma](mailto:support@example.com). Please provide a detailed description of your inquiry or issue, and we'll get back to you as soon as possible.
        """)

        st.markdown('<p class="titles">Documentation</p>', unsafe_allow_html=True)
        st.write("""
        - **Documentation:** Refer to our comprehensive documentation for guidance on using the application, explanations of algorithms, and troubleshooting tips. If you encounter any difficulties, the documentation is a valuable resource for finding solutions.
        """)

        st.markdown('<p class="titles">Stay Connected</p>', unsafe_allow_html=True)
        st.write("""
        Follow us on social media for updates, announcements, and additional resources:
        - [Twitter](#insert_twitter_url)
        - [LinkedIn](#insert_linkedin_url)
        - [Facebook](#insert_facebook_url)
        """)

        st.write("""
        Thank you for being part of our user community. Your feedback and support contribute to the continuous improvement of our application.
        """)
