# CIC-IDS-2017-Intrusion-Detection

This repository contains a comprehensive machine learning pipeline for **Intrusion Detection System (IDS)** using the **CIC-IDS-2017 dataset**. The project demonstrates data loading, preprocessing, training and evaluation of several popular machine learning models (Random Forest, XGBoost, Isolation Forest, and Autoencoder), hyperparameter tuning, and model explainability (XAI) using SHAP and LIME.

---

## 🚀 Features

* **Automated Data Handling**: Downloads, extracts, and merges the CIC-IDS-2017 dataset.
* **Robust Preprocessing**: Handles missing values, duplicates, scales features, and addresses class imbalance using `RandomUnderSampler`.
* **Multiple Model Support**:
    * **Supervised**: Random Forest, XGBoost
    * **Unsupervised/Anomaly Detection**: Isolation Forest, Autoencoder
* **Hyperparameter Tuning**: Utilizes `GridSearchCV` to find optimal parameters for supervised models, enhancing performance.
* **Comprehensive Evaluation**: Generates detailed classification reports, confusion matrices, ROC curves, and Precision-Recall curves.
* **Model Persistence**: Saves trained models for future use.
* **Explainable AI (XAI)**: Integrates SHAP and LIME to provide insights into model predictions, helping to understand feature importance.
* **Google Colab Integration**: Optimized for seamless execution in Google Colab with Google Drive linking for persistent storage.

---

## 📂 Project Structure

```plaintext
.
├── experiment_results_single_script/
│   ├── models/
│   │   ├── random_forest_model.joblib
│   │   ├── xgboost_model.joblib
│   │   └── isolation_forest_model.joblib
│   ├── model_comparison_results.csv
│   ├── optimal_hyperparameters.csv
│   └── preprocessed_data.joblib
├── CICIDS2017_data/
│   └── MachineLearningCSV.zip
├── README.md
└── main_ids_script.py  # The core Python script
```

---

## 🛠️ Setup and Installation

This project is designed to run efficiently in a Google Colab environment, leveraging its free GPU/TPU resources for faster training, especially for the Autoencoder.

### Google Colab (Recommended)


## ⚙️ Configuration

The `CONFIGURATION` section at the top of the `main_ids_script.py` file allows you to control the pipeline execution:

```python
# --- Control Flags ---
RUN_DATA_LOADING_AND_PREPROCESSING = True # Set to False to load preprocessed data from disk
TRAIN_RF = True      # Train Random Forest
TRAIN_XGB = True     # Train XGBoost
TRAIN_IF = True      # Train Isolation Forest
TRAIN_AE = True      # Train Autoencoder (requires TensorFlow)
RUN_XAI = True       # Run SHAP and LIME explanations (can be slow)
RUN_HYPERPARAMETER_TUNING = True # NEW: Flag to control hyperparameter tuning (can be slow)

# --- Directory and File Paths ---
BASE_OUTPUT_DIR = "experiment_results_single_script"
MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models")
RESULTS_FILE = os.path.join(BASE_OUTPUT_DIR, "model_comparison_results.csv")
HYPERPARAMS_FILE = os.path.join(BASE_OUTPUT_DIR, "optimal_hyperparameters.csv")
```
Adjust these flags according to your needs. For instance, after the first run, you might set RUN_DATA_LOADING_AND_PREPROCESSING = False to speed up subsequent runs by loading preprocessed data.

## 📊 Results and Outputs (from a Sample Run)

Upon successful execution, the script will generate the following:

### 🖥 Console Output

Detailed logs about:

- Data loading  
- Preprocessing steps  
- Training times  
- Evaluation metrics for each model  

### 📈 Plots

- ROC Curves and Precision-Recall Curves for each evaluated model  
- Confusion Matrix heatmaps for each evaluated model  
- SHAP summary plots for feature importance (if `RUN_XAI` is True)  

### 📁 Saved Files (in `experiment_results_single_script/`)

- `preprocessed_data.joblib`: The processed training and testing datasets, scaler, and feature names  
- `model_comparison_results.csv`: A CSV file containing a comparison table of all trained models' performance metrics  
- `optimal_hyperparameters.csv`: A CSV file listing the optimal hyperparameters found for each model during tuning  
- `models/`: Directory containing saved `.joblib` files for each trained scikit-learn model and the Autoencoder model (if TensorFlow is available)  

---

### Model Comparison Table

| Model           | Accuracy  | Precision | Recall   | F1-Score | ROC-AUC  | FPR     |
|-----------------|-----------|-----------|----------|----------|----------|---------|
| Random Forest   | 0.996647  | 0.988687  | 0.991489 | 0.990086 | 0.999678 | 0.002305|
| XGBoost         | 0.953016  | 0.803601  | 0.955278 | 0.872900 | 0.993242 | 0.047443|
| Isolation Forest| 0.779854  | 0.216904  | 0.116260 | 0.151381 | 0.644115 | 0.085296|
| Autoencoder     | 0.888346  | 0.703589  | 0.585608 | 0.639200 | 0.905989 | 0.050134|

| Model           | TP      | FP      | TN      | FN      |
|-----------------|---------|---------|---------|---------|
| Random Forest   | 126635  | 1449    | 627069  | 1087    |
| XGBoost         | 122010  | 29819   | 598699  | 5712    |
| Isolation Forest| 14849   | 53610   | 574908  | 112873  |
| Autoencoder     | 74795   | 31510   | 597008  | 52927   |

| Model           | Inference Time (s/1k samples) | Training Time (s) |
|-----------------|-------------------------------|-------------------|
| Random Forest   | 0.007236                      | 2021.701060       |
| XGBoost         | 0.003010                      | 247.218300        |
| Isolation Forest| 0.012708                      | 3.930147          |
| Autoencoder     | 0.006109                      | 68.433708         |

> **Note:** ROC-AUC for unsupervised models like Isolation Forest and Autoencoder are calculated using their anomaly scores, treating higher scores as indicative of the positive class.

---

### Optimal Hyperparameter Settings

| Model            | Key Hyperparameter                      | Optimal Value Found |
|------------------|----------------------------------------|---------------------|
| Random Forest    | max_depth                              | 10                  |
| Random Forest    | min_samples_split                      | 2                   |
| Random Forest    | n_estimators                          | 50                  |
| XGBoost          | learning_rate                         | 0.05                |
| XGBoost          | max_depth                            | 3                   |
| XGBoost          | n_estimators                         | 50                  |
| Isolation Forest | bootstrap                            | False               |
| Isolation Forest | contamination                        | 0.1                 |
| Isolation Forest | max_features                        | 1.0                 |
| Isolation Forest | max_samples                         | auto                |
| Isolation Forest | n_estimators                        | 100                 |
| Isolation Forest | n_jobs                              | -1                  |
| Isolation Forest | random_state                        | 42                  |
| Isolation Forest | verbose                            | 0                   |
| Isolation Forest | warm_start                         | False               |
| Autoencoder      | latent_dim                         | 8                   |
| Autoencoder      | optimizer                         | adam                |
| Autoencoder      | epochs                            | 50                  |
| Autoencoder      | batch_size                        | 512                 |
| Autoencoder      | reconstruction_threshold_percentile | 0.000445            |

---

## 💡 Explainable AI (XAI) Insights

The script integrates two popular XAI techniques:

- **SHAP (SHapley Additive exPlanations):**  
  Provides a global understanding of feature importance and how each feature contributes to the model's output. The summary plot helps visualize the impact of features across the dataset.

- **LIME (Local Interpretable Model-agnostic Explanations):**  
  Offers local explanations for individual predictions, showing which features are most influential for a specific instance's classification.
