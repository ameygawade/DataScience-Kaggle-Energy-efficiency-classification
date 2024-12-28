# Multi-Output Classification
 
## Classifying energy efficiency of buildings based on Heating and Cooling load

## Project Overview
This project evaluates energy efficiency in buildings using the Energy Efficiency Dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset). It employs advanced machine learning techniques to classify buildings based on their heating and cooling load demands, enabling energy optimization and personalized solutions for customers.

## What is the Business problem we are trying to solve with this project?
How can we optimize energy distribution and provide personalized energy solutions to customers by classifying buildings based on their heating and cooling load demands?

### Key Objectives
- Develop a predictive model to classify buildings' heating and cooling loads.
- Identify critical features influencing energy efficiency.
- Enable energy providers to optimize resource allocation and offer tailored energy plans.
- Contribute to sustainability by reducing energy waste.

## Dataset Details
- **Source:** [Energy Efficiency Dataset on Kaggle](https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset)
- **Sample Size:** 768 building configurations
- **Features:**
  - Relative Compactness
  - Surface Area
  - Wall Area
  - Roof Area
  - Overall Height
  - Orientation
  - Glazing Area
  - Glazing Area Distribution
- **Targets:** Heating Load and Cooling Load

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:**
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - Scikit-learn
  - XGBoost
  - skopt (for Bayesian Optimization)

## Project Workflow

### Step-1: Data Loading and Cleaning:
- Loaded the dataset from Kaggle.
- Conducted initial data checks to ensure no missing or null values.
- Renamed columns for interpretability.

### Step-2 : Exploratory Data Analysis (EDA):
- Assessed feature importance and identified significant drivers like Overall Height and Roof Area.
- Visualized data distributions and relationships through scatter plots and histograms.
- Analyzed correlations using a correlation matrix.
- Grouped Heating and Cooling Classes to define.

### Step-3 : Data Preprocessing for Multi-Output Classifier:
- Split the dataset into Training (80%) and Testing (20%) sets using train_test_split.
- Standardized features for uniform scaling using StandardScaler.

### Step-4 : Meta Model Training and Evaluation:
- Trained models using machine learning techniques with MultiOutputClassifier:
  - Support Vector Machine (SVM) 
  - Random Forest.
  - Gradient Boosting.
  - XGBoost.
- Evaluated all meta model performance using:
  - Precision, Recall, F1-Score.
  - Confusion Matrices to visualize misclassifications.

### Step-5 : Individual Model Training and Performance Evaluation for Heating Load and Cooling Load:
#### Hyperparameter Tuning:
- Fine-tuned model parameters using Bayesian Optimization for XGBoost and other advanced models for respective Output features.
- Optimized parameters such as learning rate, max depth, and subsample size to enhance performance.

#### Performance Evaluation:
- Evaluated model performance using:
  - Precision, Recall, F1-Score.
  - AUC-ROC for multi-class classification.
  - Confusion Matrices to visualize misclassifications.
- XGBoost achieved the highest accuracy of (~99%) for Heating and (~96%) for Cooling Load classifications.

### Step-6: Optimization:
- Performed dimensionality reduction while maintaining accuracy.
- Adjusted thresholds for better precision-recall balance.

### Visualization:
- Plotted actual vs predicted classes for Heating and Cooling Load.
- Generated scatter plots, heatmaps, and ROC curves for analysis and validation.

## Results and Insights
- **Model Performance:** XGBoost explained 99% of the variance in heating and cooling classifications.
- **Key Features:** Overall Height and Roof Area were critical for predicting energy demands.
- **Actionable Insight:** Targeting buildings with high energy demands can improve energy savings and efficiency.

## Limitations
- MultiOutputClassifier is not ideal for production due to scalability and resource constraints. A single unified model is preferred for deployment.

## Future Directions
- Integrating the model into energy management systems.
- Extending the dataset for broader generalization.
- Exploring the impact of environmental factors like climate zones.
