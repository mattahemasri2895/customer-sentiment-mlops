# Customer Sentiment Analysis – MLOps Dashboard

## 📌 Project Overview

The **Customer Sentiment Analysis MLOps Dashboard** is a machine learning project that analyzes customer reviews and classifies them based on sentiment. It provides a web interface for predicting sentiment, recording predictions, and visualizing sentiment analytics.

The project combines machine learning, a Flask REST API, a web frontend, and data visualization to demonstrate an end-to-end sentiment analysis workflow.

## 🎯 Objectives

- Analyze customer reviews automatically.
- Predict the sentiment of a review using a trained machine learning pipeline.
- Display prediction results through a user-friendly web interface.
- Store prediction records in a database.
- Visualize sentiment statistics through an analytics dashboard.

## ✨ Features

- **Sentiment Prediction:** Classifies customer review text using a trained ML pipeline.
- **Web Interface:** Allows users to interact with the sentiment prediction system.
- **Prediction History:** Stores prediction records for analysis.
- **Analytics Dashboard:** Displays sentiment distribution, prediction counts, and confidence statistics.
- **Data Preprocessing:** Prepares review data for machine learning.
- **Model Training:** Includes scripts for training the sentiment analysis model.
- **Data Export:** Supports exporting prediction data for further analysis.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** Scikit-learn and a trained sentiment classification pipeline
- **Data Processing:** Pandas
- **Database:** SQLite
- **Model Storage:** Joblib
- **Data Visualization:** Web-based analytics dashboard
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
customer-sentiment-mlops/
│
├── frontend/
│   └── index.html
│
├── models/
│   ├── sentiment_pipeline.joblib
│   └── metrics.json
│
├── src/
│   ├── app.py
│   ├── database.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── create_sample.py
│   └── export_data.py
│
├── check_dataset.py
├── check_processed.py
└── .gitignore
```

*Note: The raw datasets, processed datasets, and local database files are excluded from the GitHub repository to keep it lightweight. Your actual local files may include additional folders and scripts.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/mattahemasri2895/customer-sentiment-mlops.git
cd customer-sentiment-mlops
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

If the project contains a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

Otherwise, install the Python packages required by the project, based on the imports used in the source files.

### 4. Run the application

```bash
python src/app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://127.0.0.1:5000
```

## 🔄 How It Works

1. The user enters a customer review in the web interface.
2. The frontend sends the review to the Flask backend.
3. The trained machine learning pipeline processes the review and predicts its sentiment.
4. The application displays the prediction and associated confidence information, where available.
5. The prediction is recorded in the database.
6. The dashboard presents available analytics about the stored predictions.

## 📊 Dashboard

The analytics dashboard provides a summary of recorded predictions, including:

- Total reviews analyzed
- Positive, negative, and neutral sentiment counts
- Sentiment distribution
- Average prediction confidence
- Recent prediction records

The displayed statistics depend on the predictions stored in the local database.

## 🚀 Future Enhancements

- Deploy the application to a cloud platform.
- Automate model retraining and evaluation.
- Add model performance monitoring.
- Integrate continuous integration and continuous deployment (CI/CD).
- Expand the dashboard with additional evaluation metrics.
- Support larger datasets through an efficient data pipeline.

## 👩‍💻 Author

**Hemasri Matta**

GitHub: [@mattahemasri2895](https://github.com/mattahemasri2895)

## 📄 License

This project is intended for educational and portfolio purposes. Add a suitable open-source license if you plan to distribute or reuse the code publicly.
