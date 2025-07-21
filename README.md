# 🌸 Bagging Classifier Streamlit App 🔗 **[Live Demo](https://bagging-classifier-app-w99ufcaqhedzw9bnffowna.streamlit.app/)**

This project is a simple, interactive **Streamlit web application** that demonstrates how **Bagging (Bootstrap Aggregating)** improves classification performance using different base estimators like:

- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)

Built using the classic **Iris flower dataset**, this app allows users to:

- Select a base classifier  
- Tune hyperparameters like number of estimators, max samples, max features  
- Toggle bootstrap options for samples and features  
- Compare base model vs bagging model visually and quantitatively  

---

## 🧠 What is Bagging?

**Bagging** is an ensemble machine learning technique that improves accuracy and robustness by training multiple versions of a base estimator on **random subsets** of the data and aggregating their outputs (e.g., by majority voting).

---

## 📊 Features

- ✅ User-friendly Streamlit sidebar controls  
- ✅ Choose between `DecisionTree`, `KNN`, or `SVM`  
- ✅ Adjustable hyperparameters:
  - `n_estimators`
  - `max_samples`
  - `max_features`
  - `bootstrap_samples` and `bootstrap_features`
- ✅ Visual comparison of decision boundaries
- ✅ Performance metrics:
  - Accuracy scores  
  - Confusion matrices  
  - Original data scatter plot

---

## 📁 Project Structure

bagging-classifier-app/
├── app.py # Main Streamlit app
├── Iris.csv # Dataset used in the app
└── requirements.txt # Project dependencies

yaml
Copy
Edit

---

## ✅ Getting Started

### 1. Clone the repository

bash
git clone https://github.com/your-username/bagging-classifier-app.git
cd bagging-classifier-app
## 2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

## 3. Run the Streamlit app
streamlit run app.py

## 📦 Requirements
The project uses the following Python packages:

streamlit
scikit-learn
pandas
matplotlib
seaborn
Install them with:

bash
Copy
Edit
pip install -r requirements.txt
📚 Dataset Info
Name: Iris Flower Dataset

Source: UCI Machine Learning Repository

Features: Sepal length, Sepal width, Petal length, Petal width

Target: Species (Setosa, Versicolor, Virginica)

## 🛠️ Future Improvements
Add support for Bank Marketing Dataset

Add more ensemble models like Random Forest, AdaBoost

Display ROC curves and AUC scores

Export predictions as downloadable CSV files

## 🙋‍♂️ Author
Navneet Shukla
🔗 LinkedIn
🐙 GitHub

## 📃 License
This project is licensed under the MIT License.
