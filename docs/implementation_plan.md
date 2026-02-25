# Implementation Plan

Phase 1: Data Processing
- Load dataset from /data
- Clean missing values
- Encode categorical variables

Phase 2: Model Training
- Train classification model (Logistic Regression / RandomForest)
- Evaluate accuracy and save model to /models

Phase 3: Explainable AI
- Use SHAP to explain predictions
- Generate feature importance plots

Phase 4: Application Interface
- Build Streamlit app (src/app.py)
- Accept user inputs
- Show prediction + explanation

Phase 5: What-If Analysis
- Allow user to modify parameters dynamically
- Recalculate approval probability

Deliverable:
A working explainable loan decision simulator.