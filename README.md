## Description
This project applies linear regression using Stochastic Gradient Descent (SGD) to predict real estate property prices based on multiple housing attributes. 
It uses the SGDRegressor model from Scikit-learn to train and evaluate the model on a dataset of housing features such as transaction date, house age, distance 
to the nearest MRT station, number of convenience stores, and geographical coordinates. The program automates data preprocessing, model training, evaluation, and 
visualization of results — including predicted vs. actual values, residual plots, and statistical checks for model performance.

## What the Project Does
The workflow consists of three main components:
1. Data Preprocessing
- Loads the dataset from a given URL.
- Splits the data into training and test sets.
- Scales the input features using StandardScaler for better convergence.

2. Model Training
- Trains an SGDRegressor model across multiple hyperparameter configurations (alpha, max_iter, tol).
- Evaluates each configuration on training data to find the model with the lowest Mean Squared Error (MSE).
- Logs all experiment details (parameters, performance metrics, coefficients) in model2_training.log.

3. Model Evaluation & Visualization
- Evaluates the best model on test data using:
  - Mean Squared Error (MSE)
  - R² Score
  - Explained Variance Score
- Compares performance against a baseline (mean-based) predictor.
- Generates visualizations:
  - Actual vs. Predicted prices
  - Prices vs. Key feature
  - Residuals vs. Predictions
  - Histogram of residuals
  - Normal Q-Q plot to assess normality of residuals
 
## Technology Used
- Python 3.x

- Libraries:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - scipy
   - logging
 
## How to Use / Run the Program
1. Clone the Repository
git clone https://github.com/BenTennyson4/real-estate-valuation-dataset.git
cd real-estate-valuation-dataset

2. Install Dependencies
Ensure you have Python installed, then install the required libraries:
pip install pandas numpy scikit-learn matplotlib scipy

3. Run the Script
Execute the script directly:
python model2_training.py

Note:
- The program automatically downloads the dataset from the provided GitHub URL.
- It will log model details to model2_training.log in the same directory.

4. View Results

Once the program finishes:
- Check the console output for training and evaluation metrics.
- Check the plots that visualize model performance.
- View the log file (model2_training.log) for a detailed record of trials and results.
