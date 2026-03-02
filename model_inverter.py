import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ModelInverterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Inverse Model Input Finder")
        self.geometry("650x800")

        # --- Model Data ---
        self.model_bundle = None
        self.input_names = ['T_b', 'ΔT', 'Time']
        self.output_names = ['BI_t', 'BI_b', 'H_L']

        # --- Feasibility Bounds ---
        self.bounds = {
            'T_b': (150, 350),
            'ΔT': (-10, 10),
            'Time': (0, 300)
        }

        # --- UI Setup ---
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. Model Loading Section ---
        load_frame = ttk.LabelFrame(main_frame, text="1. Load Model", padding="10")
        load_frame.pack(fill=tk.X, pady=5)

        self.model_path_label = ttk.Label(load_frame, text="No model loaded.", wraplength=500)
        self.model_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        load_button = ttk.Button(load_frame, text="Load Model File", command=self.load_model)
        load_button.pack(side=tk.RIGHT)

        # --- 2. Target Outputs Section ---
        target_frame = ttk.LabelFrame(main_frame, text="2. Set Target Outputs & Weights", padding="10")
        target_frame.pack(fill=tk.X, pady=10)

        # Create headers for the target grid
        ttk.Label(target_frame, text="Prescribe", font='-weight bold').grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(target_frame, text="Output", font='-weight bold').grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(target_frame, text="Target Value", font='-weight bold').grid(row=0, column=2, padx=5, pady=5,
                                                                               sticky="ew")
        ttk.Label(target_frame, text="Weight", font='-weight bold').grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        self.prescribe_vars = {}
        self.target_entries = {}
        self.weight_entries = {}
        for i, name in enumerate(self.output_names, start=1):
            # Checkbox to decide whether to include this output in the error function
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(target_frame, variable=var)
            chk.grid(row=i, column=0, padx=5, pady=5)
            self.prescribe_vars[name] = var

            # Label for the output name
            if name == "H_L":
                target_range = " (0-1):"
            else:
                target_range = ":"
            ttk.Label(target_frame, text=f"{name}"+target_range).grid(row=i, column=1, padx=5, pady=5, sticky="w")

            # Entry for the target value
            entry = ttk.Entry(target_frame)
            entry.grid(row=i, column=2, padx=5, pady=5, sticky="ew")
            self.target_entries[name] = entry

            # Entry for the weight
            weight_entry = ttk.Entry(target_frame)
            weight_entry.insert(0, "1.0")  # Default weight
            weight_entry.grid(row=i, column=3, padx=5, pady=5, sticky="ew")
            self.weight_entries[name] = weight_entry

        target_frame.columnconfigure(2, weight=1)
        target_frame.columnconfigure(3, weight=1)

        # --- 3. Fix Inputs Section ---
        fix_frame = ttk.LabelFrame(main_frame, text="3. (Optional) Fix Input Variables", padding="10")
        fix_frame.pack(fill=tk.X, pady=10)

        self.fix_vars = {}
        self.fix_entries = {}
        for i, name in enumerate(self.input_names):
            var = tk.BooleanVar()
            if name == "Time":
                unit = " (s):"
            else:
                unit = " (°C):"
            chk = ttk.Checkbutton(fix_frame, text=f"Fix {name}"+unit, variable=var)
            chk.grid(row=i, column=0, padx=5, pady=5, sticky="w")

            entry = ttk.Entry(fix_frame)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")

            self.fix_vars[name] = var
            self.fix_entries[name] = entry
        fix_frame.columnconfigure(1, weight=1)

        # --- 4. (Optional) Set Constraints ---
        constraint_frame = ttk.LabelFrame(main_frame, text="4. (Optional) Set Constraints", padding="10")
        constraint_frame.pack(fill=tk.X, pady=10)

        self.constrain_hl_var = tk.BooleanVar(value=False)
        chk_hl = ttk.Checkbutton(constraint_frame, text="Constrain H_L (0-1):", variable=self.constrain_hl_var)
        chk_hl.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        ttk.Label(constraint_frame, text="must be >=").grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.constrain_hl_entry = ttk.Entry(constraint_frame)
        self.constrain_hl_entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        constraint_frame.columnconfigure(2, weight=1)

        # --- 5. Action Button ---
        find_button = ttk.Button(main_frame, text="Find Feasible Inputs", command=self.find_inputs)
        find_button.pack(pady=10, ipady=5)

        # --- 6. Results Section ---
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.results_text = tk.Text(results_frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def load_model(self):
        filepath = filedialog.askopenfilename(
            title="Select a joblib model file",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            self.model_bundle = joblib.load(filepath)
            required_keys = ["poly_transformer", "models", "input_names", "output_names", "selected_features"]
            if not all(key in self.model_bundle for key in required_keys):
                raise KeyError("Model file is missing required keys.")

            self.model_path_label.config(text=f"Loaded: {filepath.split('/')[-1]}")
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model file:\n{e}")
            self.model_bundle = None
            self.model_path_label.config(text="No model loaded.")

    def _get_predictions(self, x, vars_to_optimize, fixed_inputs):
        """Helper function to get all model predictions for a given input vector x."""
        current_inputs = fixed_inputs.copy()
        for i, name in enumerate(vars_to_optimize):
            current_inputs[name] = x[i]

        # 1. Construct raw input array
        input_vector = np.array([current_inputs[name] for name in self.input_names]).reshape(1, -1)

        # 2. Transform into full polynomial feature set
        poly = self.model_bundle["poly_transformer"]
        feat_vals = poly.transform(input_vector)
        feat_names = poly.get_feature_names_out(self.input_names)

        # 3. Create a DataFrame and add the constant term
        df_poly = pd.DataFrame(feat_vals, columns=feat_names)
        df_poly.insert(0, 'const', 1.0)

        predictions = {}
        for name in self.output_names:
            model = self.model_bundle["models"][name]

            # 4. Filter only the features that survived pruning for THIS specific target
            subset_features = self.model_bundle["selected_features"][name]
            X_subset = df_poly[subset_features]

            # Predict using the subsetted features
            predictions[name] = model.predict(X_subset)[0]

        return predictions

    def find_inputs(self):
        if not self.model_bundle:
            messagebox.showwarning("Warning", "Please load a model file first.")
            return

        # --- Get and validate prescribed targets and weights ---
        prescribed_targets = {}
        prescribed_weights = {}
        try:
            for name in self.output_names:
                if self.prescribe_vars[name].get():
                    target_val = self.target_entries[name].get()
                    weight_val = self.weight_entries[name].get()
                    if not target_val or not weight_val:
                        raise ValueError(f"Target and Weight must be provided for selected output '{name}'.")
                    prescribed_targets[name] = float(target_val)
                    prescribed_weights[name] = float(weight_val)
            if not prescribed_targets:
                messagebox.showerror("Error", "Please select at least one output to prescribe.")
                return
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number entered for targets or weights.\nDetails: {e}")
            return

        # --- Get and validate fixed inputs ---
        fixed_inputs = {}
        try:
            for name in self.input_names:
                if self.fix_vars[name].get():
                    fixed_inputs[name] = float(self.fix_entries[name].get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the fixed input.")
            return

        # --- Get and validate H_L constraint ---
        min_hl_value = None
        if self.constrain_hl_var.get():
            try:
                min_hl_value = float(self.constrain_hl_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for the H_L constraint.")
                return

        # --- Set up the optimization problem ---
        vars_to_optimize = [name for name in self.input_names if name not in fixed_inputs]
        if not vars_to_optimize:
            messagebox.showerror("Error", "Cannot run search. All inputs are fixed.")
            return

        initial_guess = [np.mean(self.bounds[name]) for name in vars_to_optimize]
        opt_bounds = [self.bounds[name] for name in vars_to_optimize]

        # --- Define the objective function with normalized, weighted error ---
        def objective_function(x):
            predictions = self._get_predictions(x, vars_to_optimize, fixed_inputs)

            total_error = 0
            for name, target_val in prescribed_targets.items():
                prediction = predictions[name]
                weight = prescribed_weights[name]

                # Normalize the error to account for different output magnitudes
                if abs(target_val) > 1e-9:  # Avoid division by zero
                    normalized_error = ((prediction - target_val) / target_val) ** 2
                else:  # If target is zero, use absolute squared error
                    normalized_error = (prediction - target_val) ** 2

                total_error += weight * normalized_error
            return total_error

        # --- Define constraints ---
        constraints_list = []
        method = 'L-BFGS-B'  # Default method

        if min_hl_value is not None:
            method = 'SLSQP'  # Switch method to support constraints

            def hl_constraint(x):
                # Constraint is fun(x) >= 0
                predictions = self._get_predictions(x, vars_to_optimize, fixed_inputs)
                return predictions['H_L'] - min_hl_value

            constraints_list = [{'type': 'ineq', 'fun': hl_constraint}]

        # --- Run the optimization ---
        self.update_results("Searching for a solution...")
        result = minimize(
            objective_function,
            x0=initial_guess,
            bounds=opt_bounds,
            method=method,
            constraints=constraints_list
        )

        # --- Process and Display the results ---
        # Get final inputs regardless of success to show the user the endpoint
        found_inputs = fixed_inputs.copy()
        for i, name in enumerate(vars_to_optimize):
            found_inputs[name] = result.x[i]

        # Get final predictions for all outputs
        final_predictions = self._get_predictions(result.x, vars_to_optimize, fixed_inputs)

        # Build the result string
        if not result.success:
            # Optimizer failed to converge
            final_text = f"ERROR: Optimization failed to find a solution.\n(Reason: {result.message})\n\n"
        elif result.fun > 1e-1:
            # Optimizer converged, but the error is high
            final_text = f"WARNING: Found a solution, but the final error is high ({result.fun:.4f} > 0.1).\n"
            final_text += "(This may mean no inputs perfectly match your targets, or the constraints are too strict.)\n"
            final_text += f"(Termination: {result.message})\n\n"
        else:
            # Optimizer converged, and the error is low
            final_text = "SUCCESS: Found a feasible solution.\n\n"

        if min_hl_value is not None:
            final_text += f"Active Constraint: H_L >= {min_hl_value:.4f}\n\n"

        final_text += "--- Optimized Inputs ---\n"
        for name, val in found_inputs.items():
            final_text += f"  - {name}: {val:.4f}\n"

        final_text += f"\nFinal Weighted Normalized Error: {result.fun:.6f}\n"

        final_text += "\n--- Output Breakdown ---\n"
        for name in self.output_names:
            achieved = final_predictions[name]
            if name in prescribed_targets:
                target = prescribed_targets[name]
                if abs(target) > 1e-9:
                    ind_error = abs((achieved - target) / target)
                    final_text += f"  - {name}: Target={target:.4f}, Achieved={achieved:.4f} (Error: {ind_error:.2%})\n"
                else:
                    ind_error = abs(achieved - target)
                    final_text += f"  - {name}: Target={target:.4f}, Achieved={achieved:.4f} (Abs Error: {ind_error:.4f})\n"

            else:
                final_text += f"  - {name} (Not prescribed): Achieved Value = {achieved:.4f}\n"

        self.update_results(final_text)

    def update_results(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)
        self.update_idletasks()


if __name__ == "__main__":
    app = ModelInverterApp()
    app.mainloop()

