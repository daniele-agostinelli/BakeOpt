# Baking Analysis & Inverse Modeling Toolkit
This repository contains a two-part Python toolkit designed to analyze baking processes, fit predictive models, and perform inverse optimization to find the ideal baking parameters for specific desired outcomes.

There are two main Tkinter-based graphical user interfaces (GUIs):
- Model Fitter & Analyzer (model_fitter_analyzer.py)
- Inverse Model Input Finder (model_inverter.py)

## Model Fitter & Analyzer
- Data Processing: Automatically calculates derived metrics like Humidity Loss ($H_L$) and Temperature Difference ($\Delta T$) from raw CSV data. 
- Polynomial Regression: Fits Linear (1st order), Quadratic (2nd order), and Cubic (3rd order) models using statsmodels.
- Feature Pruning: Optional backward elimination based on p-values to prune insignificant terms from the polynomial models.
- Visualization: Generates dynamic Contour plots (Response Surfaces), Predicted vs. Actual plots, and Pearson/Spearman correlation heatmaps.
- Equation Rendering: Displays the fitted mathematical equations and $R^2$ metrics dynamically.
- Model Export: Saves the fitted models, polynomial transformers, and selected features into a .joblib bundle for later use.

<img width="1400" height="938" alt="Fitter_contour" src="https://github.com/user-attachments/assets/50212f3c-e717-4f2a-8491-4f738eff2cc5" />
<img width="1400" height="938" alt="Fitter_pred_vs_real" src="https://github.com/user-attachments/assets/330741d3-236a-48dd-bbad-9b9e05d2f1ad" />

## Inverse Model Input Finder
- Optimization: Uses scipy.optimize.minimize (L-BFGS-B or SLSQP) to reverse-engineer the required inputs (Bottom Temperature, ΔT, Time) to achieve user-prescribed target outputs (Top/Bottom Browning Index, Humidity Loss).
- Customizable Weights: Assign specific weights to different target outputs to prioritize certain baking outcomes over others.
- Parameter Fixing: Lock specific input variables (e.g., fixing the baking time) while optimizing the others.
- Constraints: Apply inequality constraints, such as ensuring Humidity Loss stays above a certain threshold.

<img width="649" height="975" alt="Model_inverter" src="https://github.com/user-attachments/assets/8a35c7ac-e89b-490e-a66f-ab20e4ae929e" />

## 🛠️ Prerequisites
These applications were developed and tested with Python 3.12 on Ubuntu 24.04 and libraries as detailed in requirements.txt. 
You can install the used packages via
```bash
pip install -r requirements.txt
```

## 📂 Expected Data Format
The Model Fitter & Analyzer expects a .csv file with the following specific columns to successfully process the data:
- mass_init (Initial mass)
- mass_end_oil (Mass at the end including oil)
- mass_oil (Mass of the oil)
- temp_bot (Bottom temperature)
- temp_top (Top temperature)
- time (Baking time)
- BI_bot (Browning Index - bottom)
- BI_top (Browning Index - top)

See the provided csv file "data_test_30_07_25.csv" for our experimental data.

## 📖 How to Use
### Step 1: Train and Export the Model
Run python model_fitter_analyzer.py. Click Import CSV File and load your dataset.Select your desired Model Type (Order 1, 2, or 3) and choose whether to apply p-value Pruning. Click Fit. Explore the "Analysis & Visualization" panel to review model accuracy and feature correlations. Click Export Model to save your trained pipeline as a .joblib file.
### Step 2: Find Feasible Inputs (Inverse Modeling) 
Run python model_inverter.py. Click Load Model File and select the .joblib file you exported in Step 1. In the "Target Outputs" section, check the boxes for the outputs you want to control, set your target values, and assign weights (higher weight = higher priority for the optimizer). (Optional) Fix specific inputs or set a strict Humidity Loss ($H_L$) constraint. Click Find Feasible Inputs. The tool will display the optimal parameters required to achieve your baking goals, alongside the projected errors.
