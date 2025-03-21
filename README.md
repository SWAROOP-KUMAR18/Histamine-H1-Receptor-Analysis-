# "Histamine H1 Receptor Analysis (CHEMBL231): Application of Lipinski's Rule, pIC50 Conversion, and Machine Learning Models"
This project focuses on analyzing the Histamine H1 receptor (CHEMBL231) from the ChEMBL database. The analysis includes the application of Lipinski's Rule, pIC50 Conversion, and the use of multiple machine learning models to predict and analyze receptor-ligand interactions.

Project Overview
The project involves the following key steps:

Data Preprocessing: Preparing the dataset from the ChEMBL database for analysis.
Lipinski's Rule Application: Assessing the drug-like properties of the compounds based on Lipinski's Rule.
pIC50 Conversion: Converting IC50 values into pIC50 to normalize and scale them for model training.
Machine Learning Models: Building and evaluating multiple machine learning models for predicting the binding affinity of ligands to the Histamine H1 receptor.
Machine Learning Models Used:
Logistic Regression
DecisionTreeClassifier
RandomForestClassifier
K-Nearest Neighbors (KNN)
LazyRegressor
These models are used to predict the binding affinity of ligands by classifying or regressing the pIC50 values based on the input features extracted from the dataset.

Prerequisites
To run this project, you will need the following Python packages:

pandas
numpy
scikit-learn
matplotlib
seaborn
ChemTools (or any package you used for data manipulation)
You can install the required dependencies using pip:
pip install pandas numpy scikit-learn matplotlib seaborn
Dataset
The data is sourced from CHEMBL231 in the ChEMBL database. The dataset contains information about various ligands and their binding affinity to the Histamine H1 receptor, represented as IC50 values.

Key Features:
Lipinski's Rule of Five: The dataset is filtered to ensure the compounds meet the drug-like criteria.
pIC50 Conversion: IC50 values are converted to pIC50 values to make them suitable for machine learning modeling.
Model Evaluation: The models are evaluated using performance metrics like accuracy, precision, recall, F1-score, and RMSE (Root Mean Squared Error) for regression tasks.

Results
The results section will include the performance of each machine learning model. The models' predictions can be visualized with appropriate graphs and metrics.

Acknowledgements
ChEMBL dataset for providing valuable data for drug discovery.
Scikit-learn, for providing machine learning models and tools for model evaluation.
RDKit for cheminformatics.
