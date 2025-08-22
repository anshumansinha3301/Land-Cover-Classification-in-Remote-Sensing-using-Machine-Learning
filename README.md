# Land-Cover-Classification-in-Remote-Sensing-using-Machine-Learning

## Overview
This project implements a machine learning pipeline to classify land cover types based on simulated remote sensing satellite data. It leverages Python libraries such as pandas, numpy, scikit-learn, seaborn, matplotlib, shap, and joblib.

## Features
- Synthetic generation of multispectral satellite data and ancillary environmental variables.
- Calculation of vegetation indices: NDVI (Normalized Difference Vegetation Index) and SAVI (Soil Adjusted Vegetation Index).
- Classification into four classes: Greenery, Water, Built-up, and Barren land.
- Comparison of three classifiers: Random Forest, Gradient Boosting, and Support Vector Machine (SVM).
- Model explanation using SHAP values to understand feature impacts.
- Visualizations of feature distributions, correlations, class distribution, and model accuracies.
- Saving the best-performing model and scaler for deployment or further analysis.

## Installation
Make sure Python 3.x is installed. Install required packages using pip:

