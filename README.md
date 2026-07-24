# Breast Cancer Diagnosis Predictor

A machine learning web app that predicts whether a breast tumor is **benign** or **malignant** from digitized cell nuclei measurements, built on the classic Wisconsin Diagnostic Breast Cancer (WDBC) dataset and deployed with Streamlit.

I built this project to get hands-on with a full ML workflow — from raw tabular data to a deployed, interactive app — using a dataset that's small enough to iterate on quickly but realistic enough to actually matter.


##  Overview

Breast cancer diagnosis today relies heavily on a pathologist visually assessing cell nuclei from a fine needle aspirate (FNA) of a breast mass. The WDBC dataset digitizes that process: each sample describes ten physical characteristics of a cell nucleus (radius, texture, perimeter, etc.), each summarized by its **mean**, **standard error**, and **"worst" (largest) value** — 30 features in total, computed for each of 569 patient samples.

This project trains a classifier on those features to predict whether a mass is benign or malignant, then wraps the trained model in a Streamlit interface so it can be used interactively rather than only from a notebook.


## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — originally provided by the University of Wisconsin
- **Samples:** 569 (357 benign, 212 malignant)
- **Features:** 30 numeric features derived from digitized images of FNA biopsies, grouped into three sets of 10:
  - **Mean** — average value across the cell nuclei in the image
  - **Standard Error (SE)** — variability across the cell nuclei
  - **Worst** — mean of the three largest/most severe values
- **Base measurements:** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- **Target:** Diagnosis — `M` (malignant) or `B` (benign)


##  Project Workflow

The notebook (`Breast_cancer_analysis.ipynb`) follows a standard supervised learning pipeline:

1. **Data loading & cleaning** — reading the raw dataset (`wdbc`), dropping identifier columns, and encoding the diagnosis label.
2. **Exploratory Data Analysis (EDA)** — using `pandas`, `matplotlib`, and `seaborn` to understand feature distributions, class balance, and correlations between features.
3. **Preprocessing** — scaling all features with a `StandardScaler` so no single measurement (e.g. area, which has a much larger numeric range than smoothness) dominates the model.
4. **Dimensionality reduction** — applying `PCA` to compress the 30 correlated features into a smaller set of uncorrelated components, reducing noise and redundancy before training.
5. **Model training** — fitting a **Support Vector Machine (SVM)** classifier on the PCA-transformed data.
6. **Evaluation** — assessing the trained model on a held-out test set.
7. **Serialization** — saving the fitted `scaler`, `pca`, and `svm_model` as `.pkl` files so the trained pipeline can be reused without retraining.
8. **Deployment** — wrapping the saved pipeline in a Streamlit app (`app.py`) for interactive predictions.


##  Model Performance

To evaluate the effectiveness of the proposed approach, two machine learning algorithms—K-Nearest Neighbors (KNN) and Support Vector Machine (SVM)—were trained on the preprocessed dataset. Before training, the data was standardized using **StandardScaler** and transformed using **Principal Component Analysis (PCA)** to reduce dimensionality while preserving most of the important information.

The **K-Nearest Neighbors (KNN)** classifier achieved an accuracy of **95.61%** with a **ROC-AUC score of 0.984**, demonstrating strong performance in distinguishing between benign and malignant tumors.

The **Support Vector Machine (SVM)** classifier delivered the best overall results, achieving an accuracy of **96.49%** and a **ROC-AUC score of 0.995**. These results indicate that the model is highly effective at separating the two classes and provides excellent predictive performance.

Based on these evaluation metrics, the **Support Vector Machine (SVM)** was selected as the final model and deployed in the Streamlit web application to provide real-time breast cancer predictions.


##  Tech Stack

- **Python** — core language
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization and EDA
- **scikit-learn** — preprocessing (`StandardScaler`), dimensionality reduction (`PCA`), and modeling (`SVM`)
- **Streamlit** — interactive web app for deployment
- **Plotly** — interactive charts (confidence gauge, feature comparison radar) in the app


##  Deployment

This is the exact link which you can use to directly interact with the model: https://senodetect-alok.streamlit.app/

##  Possible Improvements

- Compare the SVM against other classifiers (Random Forest, Logistic Regression, XGBoost) and report the best performer.
- Add cross-validation results instead of a single train/test split for a more robust performance estimate.
- Add SHAP or feature-importance visualizations so predictions are more interpretable.
- Allow batch predictions from a full CSV of patients instead of one row at a time.


##  Disclaimer

This project is for **educational and portfolio purposes only**. It is not a certified medical device and should never be used as a substitute for professional diagnosis by a qualified healthcare provider.


##  Acknowledgements

Dataset provided by the **University of Wisconsin** via the UCI Machine Learning Repository:
 Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository.
