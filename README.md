# 🌸 Bagging Classifier Streamlit App

This project is a simple, interactive Streamlit web application that demonstrates how **Bagging (Bootstrap Aggregating)** improves classification performance using different base estimators like Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).

Built using the classic **Iris flower dataset**, this app allows users to:

- Select a base classifier
- Tune hyperparameters like number of estimators, max samples, max features
- Toggle bootstrap options for samples and features
- Compare **base model** vs **bagging model** visually and quantitatively

---

## 🧠 What is Bagging?

Bagging is an ensemble machine learning technique that improves the accuracy and robustness of a model by training multiple versions of a base estimator on random subsets of data and aggregating their outputs (usually by voting for classification).

---

## 📊 Features

- User-friendly **Streamlit sidebar controls**
- Choose between `DecisionTree`, `KNN`, or `SVM`
- Adjustable:
  - `n_estimators`
  - `max_samples`
  - `max_features`
  - `bootstrap_samples` and `bootstrap_features`
- Visual comparison of **decision boundaries**
- Displays:
  - Accuracy scores
  - Confusion matrices
  - Sample scatter plot of original data

---

## 🚀 Live Demo

> Coming soon (after deployment to Streamlit Cloud)

---

## 📁 Project Structure
bagging-classifier-app/
├── app.py # Main Streamlit app
├── Iris.csv # Dataset used in the app
└── requirements.txt # Project dependencies


---

## ✅ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bagging-classifier-app.git
cd bagging-classifier-app

2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

3. Run the app
bash
Copy
Edit
streamlit run app.py


📦 Requirements
streamlit

scikit-learn

pandas

matplotlib

seaborn

You can install all with:
pip install -r requirements.txt

📚 Dataset Info
Name: Iris Flower Dataset

Source: UCI Machine Learning Repository

Features: Sepal length, Sepal width, Petal length, Petal width

Target: Species (Setosa, Versicolor, Virginica)

🛠️ Future Ideas
Add support for the Bank Marketing Dataset

Introduce Random Forest or AdaBoost models

Add performance metrics like ROC curves

Export predictions as CSV

🙋‍♂️ Author
Navneet Shukla
🔗 LinkedIn | 🐙 GitHub

📃 License
This project is licensed under the MIT License.

yaml
Copy
Edit
Let me know if you’d like a version tailored for the **Bank Marketing Dataset** or want to include a **demo video or screenshots**!

