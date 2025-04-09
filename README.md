# Titanic Survival Prediction Project

## Introduction

This project tackles the classic Kaggle challenge: predicting passenger survival aboard the RMS Titanic based on passenger data like age, class, fare, and cabin information.

The notebook walks through a complete machine learning workflow, including:
* Exploratory Data Analysis (EDA) to understand the data.
* Feature Engineering to create potentially predictive variables.
* Data Preprocessing to prepare data for modeling.
* Training, Tuning (using GridSearchCV and 5-fold Cross-Validation), and Evaluation of multiple classification models (Logistic Regression, Random Forest, XGBoost, SVC, KNN, GaussianNB).
* Experimentation with simple model ensembling (probability averaging).
* Comparison of model performance based on both cross-validation scores and Kaggle public leaderboard scores.

After evaluating the models, **Random Forest Classifier was selected** as the final model for submission because it achieved the highest score (**0.78468**) on the Kaggle public leaderboard among the tested approaches, indicating the best generalization to the competition's test data in this case.

## Files in this Repository

* **`project-titanic-prediction.ipynb`**: The main Jupyter Notebook containing all analysis, code, and commentary.
* **`titanic_submission_rfc.csv`**: (Optional: Rename if your RF submission file was different) An example submission file generated using the best-performing Random Forest model.
* **`README.md`**: This project overview.
* **`.gitignore`**: Standard Python gitignore file.
* **`LICENSE`**: Project license details : MIT

## Workflow Summary

1.  **Data Loading:** Imported standard Python libraries (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn) and loaded the `train.csv` and `test.csv` datasets.
2.  **Exploratory Data Analysis (EDA):** Performed initial data inspection (`.head()`, `.info()`, `.describe()`), identified missing values, analyzed numerical and categorical feature distributions (histograms, count plots), visualized correlations (heatmap), and explored relationships between key features and the `Survived` target (using plots with `hue='Survived'`).
3.  **Feature Engineering:** Handled missing values (`Age`, `Embarked`, `Fare`). Created new features: `Title` (from Name), `FamilySize`, `IsAlone` (from SibSp/Parch), `Cabin_First_Character`, `Cabin_Multiple` (binned), `Ticket_First_Character`. Grouped rare categories for `Title`, `Cabin_First_Character`, and `Ticket_First_Character` into `..._Group` features. Applied log transformation to `Fare`.
4.  **Preprocessing:** Applied One-Hot Encoding to final categorical features and `StandardScaler` to final numerical features. Dropped unnecessary original/intermediate columns. Separated `X_train`, `y_train`, `X_test`.
5.  **Modeling and Evaluation:**
    * Trained and tuned Logistic Regression, Random Forest, XGBoost, SVC, and KNN using `GridSearchCV` with 5-fold cross-validation, optimizing for accuracy.
    * Evaluated GaussianNB using direct 5-fold cross-validation.
    * Attempted further refinement of XGBoost parameters.
    * Tested simple probability averaging ensembles.
6.  **Model Selection & Submission:** Compared models based on mean cross-validation accuracy first and afterwards scores obtained from submitting predictions to the Kaggle public leaderboard. Selected the model with the best **leaderboard score** (Random Forest) and generated the final submission file.

## Key Findings & Results

* EDA confirmed the strong predictive potential of features like `Pclass`, `Sex`, `Age`, `Fare`, and derived `Cabin` information.
* Feature engineering successfully converted complex or missing data into usable numerical features.
* **Model Performance:**
    * Cross-validation indicated **Refined XGBoost** had the highest estimated accuracy on the training data (~0.8518).
    * However, submitting predictions to Kaggle revealed that the **tuned Random Forest model** achieved the best score on the public leaderboard (**0.78468**).
    * This common discrepancy highlights the importance of evaluating on the hold-out test set, as CV scores can sometimes be overly optimistic or reflect overfitting to the training data splits.
    * Simple ensembling techniques did not improve upon the best single model score in this instance.

## Libraries Used

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn  
* XGBoost

## How to Run

1.  Clone or download this repository.
2.  Ensure you have Python and the listed libraries installed (you might need `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`).
3.  Obtain `train.csv` and `test.csv` from the Kaggle Titanic competition page and place them in an accessible location (e.g., `/kaggle/input/titanic/` or adjust paths in the notebook).
4.  Run the cells in the `project-titanic-prediction.ipynb` notebook sequentially.
