# Necessary imports
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# To encode categorical output column
encoder = LabelEncoder()

# Set style
plt.style.use('seaborn-v0_8-bright')

# Load dataset
df = pd.read_csv('Iris.csv')

# Separating Input & Output columns
X = df.drop('Species', axis=1)
y = encoder.fit_transform(df['Species'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Helper: draw meshgrid for decision boundary
def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.05)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.05)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

# Sidebar controls
st.sidebar.markdown("# Bagging Classifier")

estimators = st.sidebar.selectbox(
    'Select base model',
    ('Decision Tree', 'KNN', 'SVM')
)

max_samples = st.sidebar.slider(
    'Max Samples', 1,
    len(X_train),
    60,
    step=5
)

# Since we are using only 2 features for training and plotting
n_input_features = 2
max_features = st.sidebar.slider(
    'Max Features',
    1, n_input_features,
    min(2, n_input_features),
    key=1234
)

bootstrap_samples = st.sidebar.radio(
    'Bootstrap Samples',
    ('True', 'False')
)

bootstrap_features = st.sidebar.radio(
    'Bootstrap Features',
    ('True', 'False'),
    key=234
)

n_estimators = int(st.sidebar.number_input('Enter number of estimators', value=10, min_value=1))

# Data visualization: original scatter
st.subheader("Sample Data Distribution")
fig_data, ax_data = plt.subplots()
sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='Species', palette='rainbow', ax=ax_data)
st.pyplot(fig_data)

# Run the algorithm
if st.sidebar.button('Run Algorithm'):
    # Use only first two features for training and plotting
    X_train_2D = X_train.values[:, :2]
    X_test_2D = X_test.values[:, :2]

    # Define estimators
    if estimators == 'Decision Tree':
        estimator = DecisionTreeClassifier()
        clf = DecisionTreeClassifier(random_state=42)
    elif estimators == 'KNN':
        estimator = KNeighborsClassifier()
        clf = KNeighborsClassifier()
    else:
        estimator = SVC(kernel='linear')
        clf = SVC(kernel='linear')

    # Base model
    clf.fit(X_train_2D, y_train)
    y_pred_base = clf.predict(X_test_2D)

    # Bagging model
    bag_clf = BaggingClassifier(
        estimator=estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=(bootstrap_samples == 'True'),
        bootstrap_features=(bootstrap_features == 'True'),
        random_state=42
    )
    bag_clf.fit(X_train_2D, y_train)
    y_pred_bag = bag_clf.predict(X_test_2D)

    # Generate meshgrid for plotting
    XX, YY, input_array = draw_meshgrid(X_train_2D)
    labels_base = clf.predict(input_array)
    labels_bag = bag_clf.predict(input_array)

    # Plotting decision boundaries
    fig_base, ax_base = plt.subplots()
    ax_base.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, cmap='rainbow')
    ax_base.contourf(XX, YY, labels_base.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax_base.set_title(f"{estimators} Decision Boundary")

    fig_bag, ax_bag = plt.subplots()
    ax_bag.scatter(X_train_2D[:, 0], X_train_2D[:, 1], c=y_train, cmap='rainbow')
    ax_bag.contourf(XX, YY, labels_bag.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax_bag.set_title("Bagging Classifier Decision Boundary")

    # Display in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{estimators} Base Classifier")
        st.pyplot(fig_base)
        st.markdown(f"**Accuracy:** {accuracy_score(y_test, y_pred_base):.2%}")
        st.markdown("**Confusion Matrix**")
        st.write(confusion_matrix(y_test, y_pred_base))

    with col2:
        st.subheader("Bagging Classifier")
        st.pyplot(fig_bag)
        st.markdown(f"**Accuracy:** {accuracy_score(y_test, y_pred_bag):.2%}")
        st.markdown("**Confusion Matrix**")
        st.write(confusion_matrix(y_test, y_pred_bag))
