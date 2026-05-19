# cfDNA Cancer Detection with Explainable AI (XAI)

**Student Assistant Project | Fraunhofer HHI - Institute for Telecommunications**

---

## Quick Start (Local)

```bash
git clone https://github.com/yentureylem/cfDNA-cancer-detection-xai.git
cd cfDNA-cancer-detection-xai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open your browser: **http://localhost:8501**

---

## Overview

This project demonstrates a complete machine learning pipeline for **cell-free DNA (cfDNA)-based early cancer detection** with integrated explainable AI (XAI) capabilities. The application showcases rigorous data preprocessing, feature engineering, model development with hyperparameter optimization, comprehensive evaluation, and transparent model interpretability using SHAP.

**Key Features:**
- Realistic synthetic cfDNA dataset generation
- Exploratory Data Analysis (EDA)
- Multiple ML models with hyperparameter tuning (Logistic Regression, Random Forest, Neural Networks)
- Comprehensive model evaluation (accuracy, precision, recall, F1, ROC-AUC)
- SHAP-based explainability for model transparency
- Interactive Streamlit interface for model exploration and prediction
- Fully documented, reproducible, and version-controlled codebase

---

## Dataset

### Synthetic cfDNA Data Generation

The dataset is **synthetically generated** with **medical-realistic parameters** based on cfDNA sequencing characteristics:

**10 Features (cfDNA Biomarkers):**
1. `fragment_length_mean` - Mean cfDNA fragment length (healthy: ~167bp, cancer: ~145bp)
2. `fragment_length_std` - Standard deviation of fragment length (healthy: low, cancer: high)
3. `GC_content_ratio` - GC content percentage (altered in cancer)
4. `mapping_quality_mean` - Sequencing mapping quality
5. `coverage_depth` - Sequencing depth
6. `nucleosome_spacing` - Nucleosome spacing signature (cancer: altered)
7. `breakpoint_entropy` - Breakpoint entropy score (cancer: elevated)
8. `low_mapability_ratio` - Low mappability regions ratio
9. `tumor_fraction_estimate` - Estimated tumor fraction (cancer: detectable)
10. `dinucleotide_bias` - Dinucleotide bias score

**Class Distribution:**
- Class 0 (Healthy): Realistic healthy control characteristics
- Class 1 (Cancer): Characteristics reflecting cancer-related cfDNA alterations

**Rationale:** Synthetic data allows controlled feature engineering while maintaining medical realism, enabling validation of ML approaches before clinical deployment.

---

## Project Structure

```
cfDNA_cancer_detection/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    └── synthetic_data.csv    # Generated dataset (optional save)
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Clone Repository
```bash
git clone https://github.com/yentureylem/cfDNA-cancer-detection-xai.git
cd cfDNA-cancer-detection-xai
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Usage Guide

### Tab 1: Data Explorer
- **Class Distribution:** Visualize healthy vs. cancer samples
- **Feature Statistics:** Summary statistics for all 10 features
- **Feature Distributions:** Compare feature distributions between classes

**What it demonstrates:**
- ✅ Data exploration and understanding
- ✅ Class balance assessment
- ✅ Feature-level class separation

### Tab 2: Model Training
- Select model type: Logistic Regression, Random Forest, or Neural Network
- Automatic hyperparameter tuning via GridSearchCV (5-fold cross-validation)
- Display best parameters and cross-validation AUC

**Hyperparameter Search Spaces:**
- **Logistic Regression:** C ∈ {0.001, 0.01, 0.1, 1, 10}
- **Random Forest:** n_estimators ∈ {50, 100, 200}, max_depth ∈ {5, 10, 15}
- **Neural Network:** hidden_layers ∈ {(64,32), (128,64), (256,128)}, learning_rate ∈ {0.001, 0.01}

**What it demonstrates:**
- ✅ Rigorous hyperparameter optimization
- ✅ Cross-validation methodology
- ✅ Model selection based on validation metrics

### Tab 3: Evaluation
- **Metrics:** Accuracy, Precision, Recall, F1-Score (Train & Test)
- **Confusion Matrix:** Visual representation of predictions
- **ROC Curve:** AUC-ROC performance visualization
- **Train vs. Test Comparison:** Assess overfitting/underfitting

**What it demonstrates:**
- ✅ Comprehensive model evaluation
- ✅ Statistical rigor in performance measurement
- ✅ Generalization assessment

### Tab 4: Explainability (SHAP - Local)
- **Feature Importance:** Mean absolute SHAP values ranked by importance
- **Individual Prediction Explanation:** SHAP values for specific test samples
- **Decision Plot:** Visual explanation of how features influence prediction

**What it demonstrates:**
- ✅ Model explainability (XAI)
- ✅ Feature importance ranking
- ✅ Transparent decision-making (critical for healthcare)

*Note: SHAP requires local deployment due to computational intensity.*

### Tab 5: Prediction
- Input values for all 10 features
- Real-time prediction with confidence scores
- Probability distribution visualization

**What it demonstrates:**
- ✅ Model deployment readiness
- ✅ User-facing prediction interface

---

## Methodology

### Data Preprocessing
1. **Feature Scaling:** StandardScaler (zero mean, unit variance)
2. **Train-Test Split:** Stratified split (default 80/20)
3. **No Missing Values:** Synthetic data generation ensures completeness

### Model Development
1. **Multiple Algorithms:** Logistic Regression, Random Forest, Neural Networks
2. **Hyperparameter Optimization:** GridSearchCV with 5-fold cross-validation
3. **Scoring Metric:** ROC-AUC (suitable for imbalanced medical classification)

### Model Evaluation
- **Classification Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrix:** TP, FP, TN, FN analysis
- **Train/Test Comparison:** Generalization assessment

### Explainability (XAI)
- **SHAP (SHapley Additive exPlanations):**
  - TreeExplainer for tree-based models (Random Forest)
  - KernelExplainer for neural networks
  - Feature importance ranking
  - Individual prediction explanations (critical for regulatory compliance)

---

## Key Results & Insights

**Typical Model Performance:**
- **Best Model:** Random Forest (ensemble robustness)
- **Test Accuracy:** ~85-90%
- **Test ROC-AUC:** ~0.88-0.95
- **Precision/Recall Balance:** Configurable based on clinical decision thresholds

**Feature Importance (Typical):**
1. `tumor_fraction_estimate` - Strongest cancer indicator
2. `breakpoint_entropy` - cfDNA fragmentation pattern
3. `fragment_length_std` - Heterogeneity signature
4. `nucleosome_spacing` - Cancer-specific alteration
5. `GC_content_ratio` - Sequence composition bias

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **ML Models** | Scikit-learn |
| **Hyperparameter Optimization** | GridSearchCV |
| **Evaluation** | Scikit-learn metrics |
| **Explainability** | SHAP |
| **Visualization** | Plotly |
| **Version Control** | Git |

---

## Reproducibility

### Seeding & Randomization
- Set random seed via sidebar control
- Ensures reproducible model training and evaluation
- Critical for clinical validation and regulatory compliance

### Documentation
- Comprehensive code comments
- Feature descriptions and units
- Methodology documentation (this README)
- Clear function signatures and docstrings

### Validation
- Stratified train-test split (maintains class distribution)
- 5-fold cross-validation during hyperparameter tuning
- Test set performance as final validation metric

---

## Clinical Considerations

**Important Limitations:**
- This is a **proof-of-concept** using synthetic data
- Real cfDNA datasets require:
  - Proper institutional review and ethical approval
  - Validation on independent clinical cohorts
  - Regulatory compliance (FDA, CE marking)
  - Clinical validation studies

**Model Interpretability (XAI Benefits):**
- SHAP values provide transparent decision reasoning
- Crucial for:
  - Regulatory approval (explainability required)
  - Clinical acceptance (physicians need to understand predictions)
  - Legal liability and accountability
  - Model debugging and improvement

---

## Future Enhancements

1. **Real Dataset Integration:** Connect to clinical cfDNA databases
2. **Advanced Architectures:** Transformer-based models for sequence data
3. **Multi-Task Learning:** Simultaneous cancer type classification
4. **Uncertainty Quantification:** Bayesian approaches for confidence intervals
5. **Model Deployment:** Docker containerization, cloud deployment (AWS, Azure)
6. **Continuous Integration:** Automated testing, model monitoring
7. **Regulatory Compliance:** FDA 21 CFR Part 11 documentation

---

## References

- SHAP: https://github.com/shap/shap
- Scikit-learn: https://scikit-learn.org
- Streamlit: https://streamlit.io
- cfDNA Research: Bettegowda et al. (2014), Ignatiadis & Dawson (2018)

---

## Author

**Eylem Yentür**  
M.Sc. Research in Media Engineering | TU Ilmenau  
GitHub: github.com/yentureylem

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

Fraunhofer HHI - Institute for Telecommunications  
Project: CaniSense Platform - cfDNA-Based Early Cancer Detection
