
# Hospitalization Risk Prediction Modeling with AutoML

Rehospitalization (hospital readmission within a short period after discharge, typically 30 days) is a critical issue in healthcare for several reasons:

1. **High Costs for Hospitals and Payers**  
   - Readmissions increase healthcare spending, costing billions annually.
   - Hospitals face financial penalties (e.g., Medicare’s Hospital Readmissions Reduction Program) for excessive readmission rates.

2. **Indicator of Care Quality**  
   - High readmission rates may indicate premature discharges, inadequate follow-up, or gaps in care coordination.
   - Reducing readmissions improves care continuity, patient safety, and health outcomes.

3. **Impacts Hospital Rankings and Reputation**  
   - Readmission rates are publicly reported and factor into hospital quality ratings.
   - Hospitals with low readmission rates are viewed as providing higher-quality care.

4. **Patient and Family Burden**  
   - Readmissions expose patients to risks like infections and medication errors.
   - They cause psychological, financial, and caregiving burdens on patients and families.

5. **Often Preventable**  
   - Many readmissions are avoidable with better follow-up, medication management, and transitional care.
   - Predictive models can help hospitals identify high-risk patients and intervene proactively.

Reducing rehospitalizations is a key target for improving healthcare quality, reducing costs, and enhancing patient outcomes.

This project uses an AutoML pipeline to predict hospitalization risk from patient data. It automates data loading, preprocessing, model training, hyperparameter tuning, and evaluation using ROC AUC and Precision-Recall metrics.

## Project Workflow

1. **Import Libraries**: Load necessary Python packages (scikit-learn, Keras, etc.).
2. **Load Dataset**: Read input dataset into pandas DataFrame.
3. **Data Preprocessing**: Handle missing values, encode categorical variables, normalize features.
4. **Exploratory Data Analysis**: Generate summary statistics and visualizations.
5. **Feature Engineering**: Select and transform features for modeling.
6. **Train/Test Split**: Partition dataset into training and test sets.
7. **AutoML Model Definition**: Define a Keras model creation function for hyperparameter tuning.
8. **Hyperparameter Tuning**: Use KerasTuner Hyperband to find best model parameters optimizing validation precision.
9. **Model Evaluation**: Evaluate trained model on test set (loss, precision, recall).
10. **ROC and PR Curve Plotting**: Visualize model performance with ROC and precision-recall curves.
11. **Save Model**: Optionally export trained model to disk.
12. **Inference on New Data**: Predict hospitalization risk for unseen data.

## Dependencies

- Python >= 3.7
- tensorflow >= 2.x
- keras-tuner
- scikit-learn
- pandas
- matplotlib

Install dependencies via:

```bash
pip install tensorflow keras-tuner scikit-learn pandas matplotlib
```

## Running the Notebook

1. Open `Documented_AutoML_Hospitalization_Risk_Prediction.ipynb` in Jupyter Notebook or JupyterLab.
2. Run each cell sequentially.
3. Modify dataset paths or model parameters as needed.

## Outputs

- Best hyperparameters identified by KerasTuner
- Evaluation metrics on test data (loss, precision, recall)
- ROC and Precision-Recall curve visualizations
- Saved trained model (optional)

## Notes

- Target variable should be binary (hospitalization = 0 or 1)
- Model tuning optimizes validation precision by default



