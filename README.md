# 📊 Predictive Screen Time Manager

This is a Streamlit web application that analyzes your screen time data, provides interactive visualizations, and uses machine learning to forecast your future usage.

## 🚀 Features

* **Interactive Dashboard:** All charts are built with Plotly for hovering, zooming, and interaction.
* **Predictive ML Forecasting:** Uses a `scikit-learn` Linear Regression model to forecast your next 7 days of screen time based on historical patterns.
* **Personalized Goal Setting:** Set a daily usage goal and get alerts based on your progress.
* **In-Depth Analysis:** See breakdowns by app, rolling averages, usage distribution (pie chart), and a correlation heatmap.

---

## 🛠️ Tech Stack

* **Python:** The core language for the app.
* **Streamlit:** For building the interactive web app UI.
* **Pandas:** For data cleaning and manipulation.
* **Plotly:** For interactive data visualizations.
* **Scikit-learn:** For the time-series forecasting model.

---

## 🏃 How to Run This Project Locally

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_GITHUB_REPO_URL_HERE]
    cd [YOUR_REPO_NAME_HERE]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

5.  Upload your own screen time data (CSV) when prompted in the app's sidebar.
