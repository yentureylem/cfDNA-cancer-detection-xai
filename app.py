"""
cfDNA Cancer Detection with Explainable AI (XAI)
Fraunhofer HHI - Student Assistant ML Project
Author: Eylem Yentür
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)
import shap
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="cfDNA Cancer Detection XAI",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 cfDNA-Based Early Cancer Detection")
st.subheader("Machine Learning Model Development & Explainable AI Analysis")

# ==================== DATA GENERATION ====================
@st.cache_data
def generate_synthetic_cfDNA_data(n_samples=1000, seed=42):
    """
    Generate realistic synthetic cfDNA dataset.
    Features represent normalized cfDNA fragment characteristics.
    """
    np.random.seed(seed)
    
    # Feature names based on cfDNA characteristics
    feature_names = [
        'fragment_length_mean',      # Mean cfDNA fragment length
        'fragment_length_std',       # Std of fragment length
        'GC_content_ratio',          # GC content percentage
        'mapping_quality_mean',      # Mapping quality
        'coverage_depth',            # Sequencing depth
        'nucleosome_spacing',        # Nucleosome spacing signature
        'breakpoint_entropy',        # Breakpoint entropy score
        'low_mapability_ratio',      # Low mappability regions ratio
        'tumor_fraction_estimate',   # Estimated tumor fraction
        'dinucleotide_bias',         # Dinucleotide bias score
    ]
    
    X = np.zeros((n_samples, len(feature_names)))
    
    # Generate healthy controls (class 0)
    healthy_idx = np.arange(n_samples // 2)
    X[healthy_idx, 0] = np.random.normal(167, 8, len(healthy_idx))      # fragment length ~167bp
    X[healthy_idx, 1] = np.random.normal(22, 3, len(healthy_idx))       # low std
    X[healthy_idx, 2] = np.random.normal(0.42, 0.03, len(healthy_idx))  # GC content
    X[healthy_idx, 3] = np.random.normal(50, 5, len(healthy_idx))       # high mapping quality
    X[healthy_idx, 4] = np.random.normal(100, 15, len(healthy_idx))     # coverage
    X[healthy_idx, 5] = np.random.normal(185, 8, len(healthy_idx))      # nucleosome spacing
    X[healthy_idx, 6] = np.random.normal(3.2, 0.4, len(healthy_idx))    # low entropy
    X[healthy_idx, 7] = np.random.normal(0.05, 0.02, len(healthy_idx))  # low ratio
    X[healthy_idx, 8] = np.random.normal(0.02, 0.01, len(healthy_idx))  # very low TF
    X[healthy_idx, 9] = np.random.normal(1.0, 0.1, len(healthy_idx))    # low bias
    
    # Generate cancer cases (class 1)
    cancer_idx = np.arange(n_samples // 2, n_samples)
    X[cancer_idx, 0] = np.random.normal(145, 18, len(cancer_idx))       # shorter fragments
    X[cancer_idx, 1] = np.random.normal(35, 8, len(cancer_idx))         # higher std
    X[cancer_idx, 2] = np.random.normal(0.48, 0.05, len(cancer_idx))    # altered GC
    X[cancer_idx, 3] = np.random.normal(42, 8, len(cancer_idx))         # lower quality
    X[cancer_idx, 4] = np.random.normal(180, 40, len(cancer_idx))       # higher coverage
    X[cancer_idx, 5] = np.random.normal(165, 15, len(cancer_idx))       # altered spacing
    X[cancer_idx, 6] = np.random.normal(4.8, 0.6, len(cancer_idx))      # higher entropy
    X[cancer_idx, 7] = np.random.normal(0.18, 0.06, len(cancer_idx))    # higher ratio
    X[cancer_idx, 8] = np.random.normal(0.08, 0.04, len(cancer_idx))    # detectable TF
    X[cancer_idx, 9] = np.random.normal(1.35, 0.2, len(cancer_idx))     # higher bias
    
    # Clip to realistic ranges
    X = np.clip(X, 0, None)
    
    # Create labels
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    return pd.DataFrame(X, columns=feature_names), y

# ==================== SIDEBAR CONTROLS ====================
st.sidebar.header("⚙️ Configuration")
n_samples = st.sidebar.slider("Dataset Size", 500, 3000, 1000, 100)
test_size = st.sidebar.slider("Test Set Ratio", 0.1, 0.4, 0.2)
random_state = st.sidebar.number_input("Random State", 0, 1000, 42)

# ==================== LOAD & PREPROCESS DATA ====================
st.sidebar.header("📊 Data Processing")
X_raw, y = generate_synthetic_cfDNA_data(n_samples=n_samples, seed=random_state)

# Data split
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=test_size, random_state=random_state, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.sidebar.success(f"✅ Data loaded: {len(X_raw)} samples, {X_raw.shape[1]} features")
st.sidebar.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ==================== TAB INTERFACE ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Data Explorer",
    "🤖 Model Training",
    "📊 Evaluation",
    "🔍 Explainability (SHAP)",
    "🎯 Prediction"
])

# ==================== TAB 1: DATA EXPLORER ====================
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        class_counts = pd.Series(y).value_counts()
        fig = go.Figure(data=[
            go.Bar(x=['Healthy (0)', 'Cancer (1)'], y=class_counts.values, 
                   marker=dict(color=['#2ecc71', '#e74c3c']))
        ])
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Statistics")
        feature_stats = X_raw.describe().T
        st.dataframe(feature_stats[['mean', 'std', 'min', 'max']], use_container_width=True)
    
    st.subheader("Feature Distributions by Class")
    feature_select = st.selectbox("Select Feature", X_raw.columns)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=X_raw[X_raw.index.isin(np.where(y==0)[0])][feature_select],
                               name='Healthy', opacity=0.7, marker_color='#2ecc71'))
    fig.add_trace(go.Histogram(x=X_raw[X_raw.index.isin(np.where(y==1)[0])][feature_select],
                               name='Cancer', opacity=0.7, marker_color='#e74c3c'))
    fig.update_layout(barmode='overlay', height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: MODEL TRAINING ====================
with tab2:
    st.header("Model Development & Hyperparameter Tuning")
    
    model_type = st.radio("Select Model", 
                         ["Logistic Regression", "Random Forest", "Neural Network"])
    
    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            
            if model_type == "Logistic Regression":
                param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
                model = LogisticRegression(max_iter=1000, random_state=random_state)
                
            elif model_type == "Random Forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15]
                }
                model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
                
            else:  # Neural Network
                param_grid = {
                    'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128)],
                    'learning_rate_init': [0.001, 0.01]
                }
                model = MLPClassifier(max_iter=500, random_state=random_state, early_stopping=True)
            
            # Grid search
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            
            # Store in session
            st.session_state.best_model = best_model
            st.session_state.model_type = model_type
            
            st.success("✅ Model trained successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Best Parameters")
                for param, value in grid_search.best_params_.items():
                    st.write(f"**{param}:** {value}")
            
            with col2:
                st.subheader("CV Scores")
                cv_results = pd.DataFrame(grid_search.cv_results_)
                st.metric("Best CV AUC", f"{grid_search.best_score_:.4f}")

# ==================== TAB 3: EVALUATION ====================
with tab3:
    st.header("Model Performance Evaluation")
    
    if 'best_model' in st.session_state:
        model = st.session_state.best_model
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{accuracy_score(y_test, y_test_pred):.4f}")
        with col2:
            st.metric("Test Precision", f"{precision_score(y_test, y_test_pred):.4f}")
        with col3:
            st.metric("Test Recall", f"{recall_score(y_test, y_test_pred):.4f}")
        with col4:
            st.metric("Test F1-Score", f"{f1_score(y_test, y_test_pred):.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_test_pred)
            fig = px.imshow(cm, labels=dict(x="Predicted", y="True"),
                           x=['Healthy', 'Cancer'], y=['Healthy', 'Cancer'],
                           color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            auc = roc_auc_score(y_test, y_test_proba)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC = {auc:.4f}',
                                    line=dict(color='#3498db', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                                    line=dict(dash='dash', color='gray')))
            fig.update_layout(xlabel='False Positive Rate', ylabel='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)
        
        # Train vs Test comparison
        st.subheader("Model Performance Comparison")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Train': [
                accuracy_score(y_train, y_train_pred),
                precision_score(y_train, y_train_pred),
                recall_score(y_train, y_train_pred),
                f1_score(y_train, y_train_pred),
                roc_auc_score(y_train, y_train_proba)
            ],
            'Test': [
                accuracy_score(y_test, y_test_pred),
                precision_score(y_test, y_test_pred),
                recall_score(y_test, y_test_pred),
                f1_score(y_test, y_test_pred),
                roc_auc_score(y_test, y_test_proba)
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        st.session_state.y_test_proba = y_test_proba
    else:
        st.warning("⚠️ Please train a model first in the 'Model Training' tab")

# ==================== TAB 4: EXPLAINABILITY (SHAP) ====================
with tab4:
    st.header("Explainable AI Analysis (SHAP)")
    
    try:
        import shap
        
        if 'best_model' in st.session_state:
            model = st.session_state.best_model
            
            st.info("Computing SHAP values... This may take a moment for large datasets.")
            
            # SHAP explainer
            explainer = shap.TreeExplainer(model) if st.session_state.model_type == "Random Forest" \
                       else shap.KernelExplainer(model.predict_proba, X_train_scaled[:100])
            
            shap_values = explainer.shap_values(X_test_scaled)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Cancer class
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Importance (Mean |SHAP|)")
                feature_importance = pd.DataFrame({
                    'Feature': X_raw.columns,
                    'Importance': np.abs(shap_values).mean(axis=0)
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance, x='Importance', y='Feature', 
                            orientation='h', color='Importance')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Feature for Individual Prediction")
                sample_idx = st.slider("Select Test Sample", 0, len(X_test_scaled)-1, 0)
                
                sample_shap = shap_values[sample_idx]
                sample_features = X_test_scaled[sample_idx]
                
                explanation_data = pd.DataFrame({
                    'Feature': X_raw.columns,
                    'Value': sample_features,
                    'SHAP': sample_shap
                }).sort_values('SHAP', ascending=False, key=abs)
                
                fig = px.bar(explanation_data.head(10), x='SHAP', y='Feature', 
                            orientation='h', color='SHAP',
                            color_continuous_scale='RdBu_r', color_continuous_midpoint=0)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Decision Plot (Sample Analysis)")
            sample_idx_dp = st.slider("Select Sample for Decision Plot", 0, len(X_test_scaled)-1, 0)
            
            shap.decision_plot(explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                              shap_values[sample_idx_dp:sample_idx_dp+1],
                              X_test_scaled[sample_idx_dp:sample_idx_dp+1],
                              feature_names=list(X_raw.columns),
                              show=False)
            
            st.pyplot(plt.gcf(), use_container_width=True)
            
        else:
            st.warning("⚠️ Please train a model first in the 'Model Training' tab")
    
    except ImportError:
        st.info("🔧 **SHAP Feature Available in Local Deployment**")
        st.write("""
        SHAP (explainability) analysis requires additional dependencies that are 
        memory-intensive for cloud deployment.
        
        **To use SHAP locally:**
        ```bash
        git clone https://github.com/yentureylem/cfDNA-cancer-detection-xai.git
        cd cfDNA-cancer-detection-xai
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements-local.txt
        streamlit run app.py
        ```
        
        **What you'll get locally:**
        ✅ Feature importance rankings (mean |SHAP| values)
        ✅ Individual prediction explanations
        ✅ Decision plots for model transparency
        
        This demonstrates the app's full XAI capabilities!
        """)

# ==================== TAB 5: PREDICTION ====================
with tab5:
    st.header("Make Predictions on New Samples")
    
    if 'best_model' in st.session_state:
        model = st.session_state.best_model
        
        st.subheader("Input Features")
        user_input = {}
        
        cols = st.columns(2)
        for idx, feature in enumerate(X_raw.columns):
            with cols[idx % 2]:
                user_input[feature] = st.number_input(
                    f"{feature}",
                    value=float(X_raw[feature].mean()),
                    min_value=float(X_raw[feature].min()),
                    max_value=float(X_raw[feature].max() * 1.5)
                )
        
        if st.button("🔮 Predict"):
            # Prepare input
            user_array = np.array([user_input[f] for f in X_raw.columns]).reshape(1, -1)
            user_scaled = scaler.transform(user_array)
            
            # Predict
            prediction = model.predict(user_scaled)[0]
            probability = model.predict_proba(user_scaled)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Result")
                if prediction == 0:
                    st.success("🟢 **Likely Healthy**")
                    st.metric("Confidence", f"{probability[0]:.2%}")
                else:
                    st.error("🔴 **Likely Cancer**")
                    st.metric("Confidence", f"{probability[1]:.2%}")
            
            with col2:
                st.subheader("Probability Distribution")
                fig = go.Figure(data=[
                    go.Bar(x=['Healthy', 'Cancer'], y=probability,
                          marker=dict(color=['#2ecc71', '#e74c3c']))
                ])
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Please train a model first in the 'Model Training' tab")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
**cfDNA Cancer Detection with Explainable AI**  
Student Assistant Project | Fraunhofer HHI  
*Demonstrating: Data preprocessing, feature engineering, model optimization, hyperparameter tuning, 
evaluation metrics, and XAI (SHAP) for transparent ML in healthcare applications.*
""")
