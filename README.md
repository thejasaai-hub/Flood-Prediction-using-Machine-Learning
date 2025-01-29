# Flood Prediction using Machine Learning

This project predicts rainfall using a machine learning model trained on historical rainfall data from India. The application is built using Flask for the web interface and a Random Forest model for predictions.

## Features
- Processes historical rainfall data from 1901-2015.
- Predicts average monthly rainfall for Tamil Nadu.
- Uses a trained Random Forest Regressor model.
- Web interface built with Flask for easy user interaction.
- Provides predictions based on year and month input from the user.

## Dataset
The dataset used is `rainfall in india 1901-2015.csv`, which contains rainfall data for different regions in India over multiple years. The script extracts data specifically for Tamil Nadu and processes it for training the machine learning model.

### Data Preprocessing
- The dataset contains missing values which are handled by replacing them with the column-wise mean.
- The data is then grouped by subdivision and filtered for Tamil Nadu.
- The data is transformed into a long format where each row contains Year, Month, and the corresponding average rainfall.
- The month names are mapped to numerical values for ease of processing.
- Features (`Year`, `Month`) and target variable (`Avg_Rainfall`) are extracted for training.

## Installation
### 1. Clone the repository:
   ```sh
   git clone https://github.com/thejasaai-hub/flood-prediction.git
   cd flood-prediction
   ```
### 2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
### 3. Ensure the dataset (`rainfall in india 1901-2015.csv`) is available in the project directory.

## Model Training
### Training the Model
Run the `app.py` script to train and save the model:
```sh
python app.py
```
This script:
- Loads the rainfall dataset.
- Prepares the data for training by cleaning and transforming it.
- Splits the data into training and testing sets.
- Trains a Random Forest Regressor model with optimized hyperparameters.
- Saves the trained model as `model.pkl` using Pickle for later use.
- Provides evaluation metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Running the Web Application
Once the model is trained, you can start the Flask web application using:
```sh
python main.py
```
This will:
- Load the trained model from `model.pkl`.
- Serve a web interface where users can enter a Year and Month to predict the rainfall.
- Display the predicted rainfall value on a results page.
- The application runs on `http://127.0.0.1:5000/`.

### Web Application Workflow
1. User enters Year and Month in the form.
2. The application passes these values to the trained model.
3. The model predicts the expected rainfall for the given inputs.
4. The result is displayed on a webpage.

## File Structure
- `app.py` - Loads data, trains the model, and saves it as `model.pkl`.
- `main.py` - Flask application to serve the prediction model.
- `model.pkl` - Saved trained model.
- `rainfall in india 1901-2015.csv` - Dataset file.
- `templates/` - Contains `index.html` and `result.html` for the Flask web interface.
- `static/` - Contains CSS and JS files for styling the web interface.

## Dependencies
Install required dependencies with:
```sh
pip install numpy pandas scikit-learn flask pickle4
```
