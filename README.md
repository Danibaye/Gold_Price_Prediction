# Gold Price Prediction API and UI

## Project Overview

This project demonstrates a full machine learning lifecycle for predicting daily gold prices, from data acquisition to user-friendly deployment. The model is served as an API using FastAPI and accessible through a web interface.

## Project Goal

The goal is to build a complete machine learning application by:

- **Sourcing and preparing real-world data:** Obtaining and cleaning relevant data for gold price prediction.
- **Building and optimizing a regression model:** Developing a model to accurately forecast gold prices.
- **Deploying the model as a functional API:** Making the model accessible via a web API.
- **Providing a user-friendly front-end:** Creating a web interface for easy interaction with the API.

## 💻 Technology Stack

- **Python 3.10+:** Core programming language.
- **FastAPI:** Modern, fast (high-performance), web framework for building APIs.
- **Jinja2:** Templating engine for dynamic HTML generation.
- **scikit-learn:** Machine learning library with tools for modeling, preprocessing, and evaluation.
- **pandas:** Data manipulation and analysis.
- **joblib:** Library for serializing (saving) and deserializing (loading) Python objects, including trained models.
- **matplotlib/seaborn:** Data visualization.
- **uvicorn:** ASGI server for running the FastAPI application.

## 📂 Project Structure

gold_prediction_ui/
├── app.py # FastAPI application
├── templates/ # Directory for HTML templates
│ └── index.html # Main user interface
└── static/
├── script.js # JavaScript (Not needed now - server-side rendering)
└── style.css # CSS for styling
├── model.pkl # Serialized trained model
├── preprocessor.pkl # Serialized StandardScaler
└── README.md # This file

## ✅ Setup and Usage

### 1. Prerequisites

- Python 3.10+ installed.
- Poetry installed.
- Model and preprocessor files exist.

### 2. Installation

1.  Clone the repository:

    ```bash
    git clone <your_repository_url>
    cd gold_prediction_ui
    ```

2.  Install dependencies using Poetry:

    ```bash
    poetry install
    ```

    Install Poetry if needed: `pip install poetry`

3.  (Optional) Activate Poetry's virtual environment:

    ```bash
    poetry shell
    ```

### 3. Run the FastAPI Application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload


4. Access the User Interface
Open your web browser and go to http://localhost:8000.

5. Using the Application
Enter input values in the form.

Click "Predict Price".

The predicted gold price and any error messages are displayed.

🔑 API Endpoints
/ (GET): Renders the user interface (HTML form).

/predict (POST): Receives form data, predicts the gold price, and returns the results in HTML.

/health (GET): Simple endpoint to check if the API is running.

📝 Data Documentation
1. Source of Data
[Example: Quandl (https://www.quandl.com) for historical gold price data.]

2. License / Terms of Use
[Example: Publicly available data, no specific license required.]

3. Data Structure
Format: CSV

Organization: Tabular data, one row per day.

Date: Date of the record.

Close/Last: Closing price in USD.

Volume: Trading volume.

Open: Opening price in USD.

High: Highest price in USD.

Low: Lowest price in USD.

year: Year extracted from Date.

month: Month extracted from Date.

day: Day extracted from Date.

weekday: Weekday (0-6).

is_weekend: Weekend indicator (True/False).

4. Features
Described above. These features are used to predict the Close price.

🛠️ Data Preprocessing
Handling Missing Values: Dropped rows with missing target values.

Feature Engineering: Extracted temporal features from the Date column (year, month, day, weekday, is_weekend).

Scaling Numerical Features: Applied StandardScaler:

Scales each numerical feature to have zero mean and unit variance. This is important for algorithms that are sensitive to feature scaling.

Prevents features with large values from dominating the model.

Train-Test Split: 80/20 split with random_state=42 for reproducibility. Splitting the data into separate training and testing sets allows the model to be tested on previously unseen data, providing a more realistic evaluation of the model's performance.

⚙️ Model Details
Model Type: RandomForestRegressor

An ensemble learning method that combines multiple decision trees to make predictions. It is robust to outliers and non-linear relationships.

Hyperparameter Tuning: GridSearchCV

A technique to systematically search for the best combination of hyperparameters for a model by evaluating performance across multiple combinations.

The hyperparameters explored were:

n_estimators: Number of trees in the forest.

max_depth: Maximum depth of the trees.

min_samples_split: Minimum number of samples required to split an internal node.

min_samples_leaf: Minimum number of samples required to be at a leaf node.

Negative Mean Absolute Error was used as the scoring metric because it is robust to outliers. The goal of the grid search was to find a parameter space that minimizes this MAE.

Best Hyperparameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

📈 Evaluation Metrics
Mean Absolute Error (MAE): 4.263

The average absolute difference between the predicted and actual gold prices. Lower values indicate better performance.

R-squared (R2): 0.998

Represents the proportion of variance in the target variable that is explained by the model. Higher values (closer to 1) indicate a better fit. A value of 0.998 indicates the model is capturing nearly all the variance in the underlying data.

📦 Persisted Artifacts
model.pkl: The trained RandomForestRegressor model saved using joblib.

preprocessor.pkl: The StandardScaler object used for data scaling, also saved using joblib.

features.pkl: A list of feature names used by the model.

🚧 Potential Limitations and Future Improvements
Generalizability: The model's accuracy may degrade when applied to data outside the range of the training data (e.g., very different market conditions).

Feature Importance: Further analyzing feature importance could identify the most impactful factors driving gold prices.

Additional Features: Investigating external factors like economic indicators, news sentiment, and interest rates may improve performance.

Alternative Models: Experimenting with other regression models could result in better performance. XGBoost, Support Vector Regression, or time series models like ARIMA are candidates for investigation.

Limited Evaluation Metrics : Testing for more evaluation metrics is necessary.

📝 Documentation
Code Structure
app.py: Contains FastAPI application setup, API endpoints, model loading, and preprocessing logic.

static/index.html: Defines the HTML structure for the user interface.

static/style.css: Contains the CSS styles for the user interface.

Future Work
Implement more robust validation and error handling.

Enhance the UI with better styling and user experience.

Develop a more streamlined data preprocessing pipeline.

Explore different machine learning models and feature engineering techniques.

Implement unit and integration tests for the application.

Deploy the application to a cloud platform (e.g., Heroku, AWS) for production use.

🔗 License
[Specify your project's license. Example: MIT License]

🖋️ Author
[danial baye Gizachew]

Deadline: February 2, 2017 EC

**Key Points:**

*   **Explanations:** The text is much more descriptive in explaining the purpose of the project and the rationale behind different steps.
*   **Clarity:** Technical concepts are explained clearly and concisely.
*   **Real-World Application:** The limitations section discusses the real-world challenges of applying the model and potential solutions.
*   **Completeness:** Now the full documentation is very clear.

This version should give you a good foundation for a detailed and well-explained documentation report for your project.
```
