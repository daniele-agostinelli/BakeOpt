import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import re, joblib


# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
class Config:
    # These are the columns the script will use/create after processing the CSV
    INPUTS = {
        'T_b': [0, 400], # Temp bottom # Default ranges, will be updated from data
        'ΔT': [-10, 10], # top - bottom # temperature difference
        'Time': [0, 300]
    }
    OUTPUTS = [
        'BI_t', # Browning Index top
        'BI_b', # Browning Index bottom
        'H_L' # Humidity Loss
    ]


# =============================================================================
# --- 2. DATA MANAGEMENT CLASS ---
# =============================================================================
class DataManager:
    def __init__(self, input_names, output_names):
        self.input_names = input_names
        self.output_names = output_names
        self.raw_data = None
        self.processed_data = None

    def load_from_csv(self, filepath):
        try:
            self.raw_data = pd.read_csv(filepath)

            # --- Data Processing ---
            df = self.raw_data.copy()

            # 1. Calculate Humidity Loss
            # Ensure required columns exist before calculation
            if not all(col in df.columns for col in ['mass_init', 'mass_end_oil', 'mass_oil']):
                return False, "CSV is missing one of the required mass columns: 'mass_init', 'mass_end_oil', 'mass_oil'."
            df['H_L'] = (df['mass_init'] - (df['mass_end_oil'] - df['mass_oil'])) / df['mass_init']

            # 2. Calculate Temperature Change
            if not all(col in df.columns for col in ['temp_bot', 'temp_top', 'time']):
                return False, "CSV is missing one of the required columns: 'temp_bot', 'temp_top', 'time'."
            df['BI_b'] = df['BI_bot']
            df['BI_t'] = df['BI_top']
            df['T_b'] = df['temp_bot']
            df['ΔT'] = df['temp_top'] - df['temp_bot']
            df['Time'] = df['time']

            # 3. Select only the necessary columns for the model
            required_cols = self.input_names + self.output_names
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                return False, f"CSV processed, but missing required output columns: {', '.join(missing)}"

            self.processed_data = df[required_cols].astype(float)  # Ensure all data is numeric
            self.processed_data = self.processed_data.dropna()  # Drop rows with missing values

            if self.processed_data.empty:
                return False, "Processed data is empty after removing rows with missing values. Check your CSV."

            return True, "CSV loaded and processed successfully!"
        except FileNotFoundError:
            return False, f"Error: The file '{filepath}' was not found."
        except Exception as e:
            return False, f"An error occurred while processing the CSV: {e}"

    def get_data_ranges(self):
        if self.processed_data is None:
            return {}
        ranges = {}
        for col in self.input_names:
            ranges[col] = (self.processed_data[col].min(), self.processed_data[col].max())
        return ranges


# =============================================================================
# --- 3. MODEL TRAINING CLASS ---
# =============================================================================
class ModelTrainer:
    def __init__(self, input_names, output_names):
        self.input_names = input_names
        self.output_names = output_names
        self.poly_transformer = None
        self.models = {}
        self.scores = {}
        self.is_fitted = False

    def fit(self, data, poly_order):
        try:
            X = data[self.input_names].values
            self.poly_transformer = PolynomialFeatures(degree=poly_order, include_bias=True)
            X_poly = self.poly_transformer.fit_transform(X)

            for name in self.output_names:
                y = data[name].values.astype(float)
                model = LinearRegression()
                model.fit(X_poly, y)
                self.models[name] = model

                # Calculate and store R-squared score
                y_pred = model.predict(X_poly)
                self.scores[name] = r2_score(y, y_pred)

            self.is_fitted = True
            return True, "Models fitted successfully!"
        except Exception as e:
            self.is_fitted = False
            return False, f"Error during fitting: {e}"

    def predict(self, input_data):
        if not self.is_fitted:
            return {name: 'N/A' for name in self.output_names}
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        input_poly = self.poly_transformer.transform(input_data)
        predictions = {}
        for name in self.output_names:
            pred_values = self.models[name].predict(input_poly)
            predictions[name] = pred_values
        return predictions

    def get_model_equation(self, output_name):
        if not self.is_fitted or output_name not in self.models:
            return "Model not fitted for this output."

        model = self.models[output_name]
        #local_input_names = [name.replace('_', '_') for name in self.input_names]
        feature_names = self.poly_transformer.get_feature_names_out(self.input_names)

        intercept = model.intercept_
        coeffs = model.coef_

        latex_parts = []

        # Use a counter to break the line every few terms
        term_count = 1
        terms_per_line = 7  # Adjust this value based on your panel width

        # Start from index 1 to skip the intercept term
        temp = f"{output_name} = {intercept:.2e}"        # Start the equation with the output name and intercept term
        for i, (coeff, name) in enumerate(zip(coeffs, feature_names)):
            if i == 0:  # Skip the intercept term
                continue

            if abs(coeff) < 1e-6:
                continue

            # Format term for LaTeX (e.g., Temp\_Bottom^{2} \cdot Time)
            formatted_name = name.replace(" ", r" \cdot ")
            formatted_name = re.sub(r'\^(\d+)', r'^{\1}', formatted_name)

            sign = "-" if coeff < 0 else "+"
            temp = temp + f" {sign} {abs(coeff):.2e} \\cdot {formatted_name}"

            # reset text
            if term_count % terms_per_line == 0:
                latex_parts.append(temp)
                temp = ""

            term_count += 1
        latex_parts.append(temp)

        return latex_parts

    def export_model(self, filepath):
        """
        Save the fitted model (polynomial transformer + regression models) to a file.
        """
        if not self.is_fitted:
            return False, "Model has not been fitted yet."

        try:
            bundle = {
                "poly_transformer": self.poly_transformer,
                "models": self.models,
                "scores": self.scores,
                "input_names": self.input_names,
                "output_names": self.output_names
            }
            joblib.dump(bundle, filepath)
            return True, f"Model exported successfully to {filepath}"
        except Exception as e:
            return False, f"Error exporting model: {e}"

# =============================================================================
# --- 4. MAIN APPLICATION CLASS ---
# =============================================================================
class App(tk.Tk):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_manager = DataManager(list(config.INPUTS.keys()), config.OUTPUTS)
        self.model_trainer = ModelTrainer(list(config.INPUTS.keys()), config.OUTPUTS)
        self.current_ranges = config.INPUTS

        self.title("Pizza Baking Analysis Tool")
        self.geometry("1400x900")
        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=5)
        main_frame.rowconfigure(1, weight=1)

        # --- Top Control Frame ---
        top_controls = ttk.Frame(main_frame)
        top_controls.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)

        # 1. Data Loading
        data_frame = ttk.LabelFrame(top_controls, text="1. Data Loading", padding=10)
        data_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(data_frame, text="Import CSV File", command=self._import_csv).pack(pady=5)

        # 2. Model Fitting
        model_frame = ttk.LabelFrame(top_controls, text="2. Model Fitting", padding=10)
        model_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        self.model_type = tk.IntVar(value=2)
        ttk.Label(model_frame, text="Model Type:").pack(anchor='w')
        ttk.Radiobutton(model_frame, text="Linear (Order 1)", variable=self.model_type, value=1).pack(anchor='w')
        ttk.Radiobutton(model_frame, text="Quadratic (Order 2)", variable=self.model_type, value=2).pack(anchor='w')
        ttk.Radiobutton(model_frame, text="Cubic (Order 3)", variable=self.model_type, value=3).pack(anchor='w')
        self.fit_button = ttk.Button(model_frame, text="Fit", command=self._fit_models, state=tk.DISABLED)
        self.fit_button.pack(pady=5)
        # export button
        self.export_button = ttk.Button(model_frame, text="Export Model", command=self._export_model, state=tk.DISABLED)
        self.export_button.pack(pady=5)

        # 3. Model Equation
        equation_frame = ttk.LabelFrame(top_controls, text="3. Fitted Model Equation", padding=10)
        equation_frame.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)

        # Canvas for rendering LaTeX equation
        self.eq_fig = Figure(figsize=(5, 2), dpi=100)
        self.eq_fig.patch.set_facecolor(self.cget('bg'))  # Match window background
        self.eq_canvas = FigureCanvasTkAgg(self.eq_fig, master=equation_frame)
        self.eq_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Data Table Frame ---
        table_frame = ttk.LabelFrame(main_frame, text="Processed Data", padding=10)
        table_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        cols = self.data_manager.input_names + self.data_manager.output_names
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings')
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # --- Analysis and Visualization Frame ---
        analysis_frame = ttk.LabelFrame(main_frame, text="4. Analysis & Visualization", padding=10)
        analysis_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(1, weight=1)

        plot_controls_frame = ttk.Frame(analysis_frame)
        plot_controls_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))

        # Plot Controls
        ttk.Label(plot_controls_frame, text="Plot Output:").pack(side='left')
        self.plot_output_var = tk.StringVar()
        self.plot_output_menu = ttk.OptionMenu(plot_controls_frame, self.plot_output_var, 'Select Output',
                                               *self.config.OUTPUTS)
        self.plot_output_menu.pack(side='left', padx=5)
        self.plot_output_var.trace_add('write', self._on_plot_controls_changed)

        self.plot_type = tk.StringVar(value="Contour")
        ttk.Label(plot_controls_frame, text="Plot Type:").pack(side='left', padx=(20, 5))
        ttk.Radiobutton(plot_controls_frame, text="Contour", variable=self.plot_type, value="Contour",
                        command=self._setup_plot_ui).pack(side='left',padx=(0, 5))
        ttk.Radiobutton(plot_controls_frame, text="Predicted vs. Actual", variable=self.plot_type, value="PredVActual",
                        command=self._setup_plot_ui).pack(side='left',padx=(0, 5))
        ttk.Radiobutton(plot_controls_frame, text="Correlations", variable=self.plot_type, value="Correlations",
                        command=self._setup_plot_ui).pack(side='left',padx=(0, 5))

        # Plotting Area
        self.plot_canvas_frame = ttk.Frame(analysis_frame)
        self.plot_canvas_frame.grid(row=1, column=0, sticky='nsew')
        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Container for plot-specific controls (sliders, axis selectors)
        self.plot_specific_controls_frame = ttk.Frame(analysis_frame)
        self.plot_specific_controls_frame.grid(row=2, column=0, sticky='ew', pady=(10, 0))

    def _import_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not filepath: return

        success, msg = self.data_manager.load_from_csv(filepath)
        if success:
            self._update_treeview(self.data_manager.processed_data)
            self.current_ranges = self.data_manager.get_data_ranges()
            self.fit_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", msg)
        else:
            messagebox.showerror("Error", msg)
            self.fit_button.config(state=tk.DISABLED)

    def _update_treeview(self, df):
        self.tree.delete(*self.tree.get_children())
        for index, row in df.iterrows():
            display_values = [f"{v:.3f}" if isinstance(v, (float, np.floating)) else v for v in row.values]
            self.tree.insert("", "end", values=display_values)

    def _fit_models(self):
        if self.data_manager.processed_data is None:
            messagebox.showerror("Error", "No data loaded to fit.")
            return

        poly_order = self.model_type.get()
        success, msg = self.model_trainer.fit(self.data_manager.processed_data, poly_order)

        if success:
            messagebox.showinfo("Success", msg)
            self.export_button.config(state=tk.NORMAL)  # Enable export after fit
            if not self.plot_output_var.get() or self.plot_output_var.get() == 'Select Output':
                self.plot_output_var.set(self.config.OUTPUTS[0])
            else:
                self._setup_plot_ui()
            self._update_model_equation()
        else:
            messagebox.showerror("Error", msg)

    def _export_model(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".joblib",
                                                filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        if not filepath:
            return
        success, msg = self.model_trainer.export_model(filepath)
        if success:
            messagebox.showinfo("Success", msg)
        else:
            messagebox.showerror("Error", msg)

    def _update_model_equation(self, *args):
        output_name = self.plot_output_var.get()
        self.eq_fig.clear()

        if self.model_trainer.is_fitted and output_name and output_name != 'Select Output':
            equations = self.model_trainer.get_model_equation(output_name)
            score = self.model_trainer.scores.get(output_name, 0)
            score_text = f"$R^2 = {score:.4f}$"

            # Display score and equation on the figure
            self.eq_fig.text(0.02, 0.7, score_text, va='top', ha='left', fontsize=11)
            for i,equation in enumerate(equations):
                self.eq_fig.text(0.02, 0.45-0.10*i, f"${equation}$", va='top', ha='left', fontsize=9)

        self.eq_canvas.draw()

    def _on_plot_controls_changed(self, *args):
        self._update_model_equation()
        self._setup_plot_ui()

    def _setup_plot_ui(self, *args):
        if not self.model_trainer.is_fitted: return

        for widget in self.plot_specific_controls_frame.winfo_children():
            widget.destroy()

        plot_type = self.plot_type.get()
        if plot_type == "Contour":
            self._setup_contour_controls()
        elif plot_type == "PredVActual":
            self._draw_pred_vs_actual_plot()
        elif plot_type == "Correlations":
            self._setup_corr_controls()
            self._draw_correlations()

    def _setup_contour_controls(self, *args):
        frame = self.plot_specific_controls_frame
        input_names = self.data_manager.input_names

        # Clear previous widgets properly before redrawing
        for widget in frame.winfo_children():
            widget.destroy()

        controls_frame = ttk.Frame(frame)
        controls_frame.pack(fill=tk.X, expand=True, pady=5)

        ttk.Label(controls_frame, text="X-Axis:").pack(side='left', padx=(0, 5))
        if not hasattr(self, 'x_axis_var'):
            self.x_axis_var = tk.StringVar(value=input_names[0])
        self.x_axis_menu = ttk.OptionMenu(controls_frame, self.x_axis_var, self.x_axis_var.get(), *input_names,
                                          command=lambda *_: self._setup_contour_controls())
        self.x_axis_menu.pack(side='left', padx=(0, 15))

        ttk.Label(controls_frame, text="Y-Axis:").pack(side='left', padx=(0, 5))
        if not hasattr(self, 'y_axis_var'):
            self.y_axis_var = tk.StringVar(value=input_names[1])
        self.y_axis_menu = ttk.OptionMenu(controls_frame, self.y_axis_var, self.y_axis_var.get(), *input_names,
                                          command=lambda *_: self._setup_contour_controls())
        self.y_axis_menu.pack(side='left')

        self.plot_sliders = {}
        sliders_frame = ttk.Frame(frame)
        sliders_frame.pack(fill=tk.X, expand=True, pady=5)

        x_axis, y_axis = self.x_axis_var.get(), self.y_axis_var.get()
        fixed_vars = [v for v in input_names if v not in [x_axis, y_axis]]

        for var in fixed_vars:
            min_val, max_val = self.current_ranges[var]
            ttk.Label(sliders_frame, text=f"Fix {var}:").pack(side='left', padx=(10, 2))
            slider_var = tk.DoubleVar(value=np.mean([min_val, max_val]))
            slider = ttk.Scale(sliders_frame, from_=min_val, to=max_val, variable=slider_var, orient='horizontal',
                               command=self._draw_contour_plot)
            slider.pack(side='left', padx=(0, 10), fill='x', expand=True)
            self.plot_sliders[var] = slider_var

        self._draw_contour_plot()

    def _draw_contour_plot(self, *args):
        x_name, y_name = self.x_axis_var.get(), self.y_axis_var.get()
        output_name = self.plot_output_var.get()

        if not all([x_name, y_name, output_name, output_name != 'Select Output']):
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if x_name == y_name:
            ax.text(0.5, 0.5, "Please select different X and Y axes.", ha='center', va='center', fontsize=12)
            self.canvas.draw()
            return

        x_range = np.linspace(*self.current_ranges[x_name], 40)
        y_range = np.linspace(*self.current_ranges[y_name], 40)
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        input_df = pd.DataFrame(columns=self.data_manager.input_names)
        input_df[x_name] = grid_x.ravel()
        input_df[y_name] = grid_y.ravel()
        for var, slider_var in self.plot_sliders.items():
            input_df[var] = slider_var.get()

        predictions = self.model_trainer.predict(input_df[self.data_manager.input_names].values)
        z_values = predictions[output_name].reshape(grid_x.shape)

        contour = ax.contourf(grid_x, grid_y, z_values, 20, cmap='viridis', alpha=0.8)

        # Filter and overlay actual data points that are "close" to the slider's value
        filtered_data = self.data_manager.processed_data.copy()
        for var, slider_var in self.plot_sliders.items():
            slider_val = slider_var.get()
            min_r, max_r = self.current_ranges[var]
            range_width = max_r - min_r
            tolerance = range_width * 0.05 if range_width > 0 else 0.1  # 5% tolerance
            filtered_data = filtered_data[np.isclose(filtered_data[var], slider_val, atol=tolerance)]

        if not filtered_data.empty:
            ax.scatter(filtered_data[x_name], filtered_data[y_name], c=filtered_data[output_name],
                       edgecolor='k', cmap='viridis', vmin=z_values.min(), vmax=z_values.max(),
                       label='Actual Data (nearby)')

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        title_parts = [f"Response Surface for '{output_name}'"]
        for var, slider_var in self.plot_sliders.items():
            title_parts.append(f"{var}={slider_var.get():.1f}")
        ax.set_title("\n".join(title_parts), fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        #else:
        #    print("No artists with labels found to put in legend.")

        self.fig.colorbar(contour, ax=ax, label=output_name)
        self.canvas.draw()

    def _draw_pred_vs_actual_plot(self, *args):
        frame = self.plot_specific_controls_frame
        # Clear previous widgets properly before redrawing
        for widget in frame.winfo_children():
            widget.destroy()

        output_name = self.plot_output_var.get()
        if not self.model_trainer.is_fitted or not output_name or output_name == 'Select Output':
            return

        data = self.data_manager.processed_data
        X_actual = data[self.data_manager.input_names].values
        y_actual = data[output_name].values

        predictions = self.model_trainer.predict(X_actual)
        y_pred = predictions[output_name]

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        ax.scatter(y_actual, y_pred, edgecolor='k', alpha=0.7)

        # Correctly calculate limits for y=x line
        min_val = np.min([y_actual.min(), y_pred.min()])
        max_val = np.max([y_actual.max(), y_pred.max()])
        margin = (max_val - min_val) * 0.05
        lims = [min_val - margin, max_val + margin]

        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Fit')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Predicted vs. Actual for {output_name}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        self.canvas.draw()

    def _setup_corr_controls(self, *args):
        frame = self.plot_specific_controls_frame
        corr_types = ["Pearson", "Spearman"]
        corr_labels = self.config.OUTPUTS.copy()
        for a in self.config.INPUTS.keys():
            corr_labels.append(a)
        # Clear previous widgets properly before redrawing
        for widget in frame.winfo_children():
            widget.destroy()

        controls_frame = ttk.Frame(frame)
        controls_frame.pack(fill=tk.X, expand=True, pady=5)

        # SELECT CORRELATION TYPE
        ttk.Label(controls_frame, text="Correlation type:").pack(side='left', padx=(0, 5))
        # Initialize corr_type only once
        if not hasattr(self, 'corr_type'):
            self.corr_type = tk.StringVar(value=corr_types[0])
        # Create the dropdown menu
        self.corr_menu = ttk.OptionMenu(
            controls_frame,
            self.corr_type,
            self.corr_type.get(),
            *corr_types,
            command=lambda selection: self._on_corr_type_change(selection)
        )
        self.corr_menu.pack(side='left', padx=(0, 15))

        # SELECT VARIABLE 1 for correlation
        ttk.Label(controls_frame, text="X:").pack(side='left', padx=(0, 5))
        if not hasattr(self, 'corr_x'):
            self.corr_x = tk.StringVar(value=corr_labels[0])
        self.corr_x_menu = ttk.OptionMenu(controls_frame, self.corr_x, self.corr_x.get(), *corr_labels,
                                          command=lambda *_: self._setup_corr_controls())
        self.corr_x_menu.pack(side='left', padx=(0, 15))

        # SELECT VARIABLE 2 for correlation
        ttk.Label(controls_frame, text="Y:").pack(side='left', padx=(0, 5))
        if not hasattr(self, 'corr_y'):
            self.corr_y = tk.StringVar(value=corr_labels[1])
        self.corr_y_menu = ttk.OptionMenu(controls_frame, self.corr_y, self.corr_y.get(), *corr_labels,
                                          command=lambda *_: self._setup_corr_controls())
        self.corr_y_menu.pack(side='left', padx=(0, 15))

        # SELECT VARIABLE 3 for correlation
        ttk.Label(controls_frame, text="Z:").pack(side='left', padx=(0, 5))
        if not hasattr(self, 'corr_z'):
            self.corr_z = tk.StringVar(value=corr_labels[2])
        self.corr_z_menu = ttk.OptionMenu(controls_frame, self.corr_z, self.corr_z.get(), *corr_labels,
                                          command=lambda *_: self._setup_corr_controls())
        self.corr_z_menu.pack(side='left', padx=(0, 15))

        #print("Current correlation type:", self.corr_type.get())
        self._draw_correlations()
        return

    def _on_corr_type_change(self, selection):
        """Callback when the user selects a correlation type."""
        self.corr_type.set(selection)  # update StringVar explicitly
        #print("Correlation type changed to:", selection)
        self._draw_correlations()

    def _draw_correlations(self, *args):
        #corr_type = "Pearson"
        labels = [self.corr_x.get(),self.corr_y.get(),self.corr_z.get()]
        #print(labels)
        data = self.data_manager.processed_data
        X = data[labels].values
        #print(X)
        variables = [np.asarray(X[:,0]), np.asarray(X[:,1]), np.asarray(X[:,2])]
        n = len(variables)

        # Empty DataFrames for correlations and p-values
        corr = pd.DataFrame(np.zeros((n, n)), columns=labels, index=labels)
        pval = pd.DataFrame(np.zeros((n, n)), columns=labels, index=labels)

        # Compute pairwise correlations
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr.iloc[i, j] = 1.0
                    pval.iloc[i, j] = 0.0
                else:
                    if self.corr_type.get() == "Pearson":
                        r, p = pearsonr(variables[i], variables[j])
                    elif self.corr_type.get() == "Spearman":
                        r, p = spearmanr(variables[i], variables[j])
                    corr.iloc[i, j] = r
                    pval.iloc[i, j] = p

        # Create annotation matrices with correlation + p-value
        def make_annotations(corr, pval):
            return corr.round(2).astype(str) + "\n(" + pval.map(lambda v: f"{v:.2g}") + ")"

        annot = make_annotations(corr, pval)

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        sns.heatmap(corr.astype(float), annot=annot, fmt="", cmap="coolwarm",
                    center=0, cbar_kws={"label": "Correlation"}, ax=ax)
        ax.set_title(f"{self.corr_type.get()} Correlation (r, p-value)")

        self.canvas.draw()

        return {"corr": corr, "pval": pval}

if __name__ == "__main__":
    app = App(Config)
    app.mainloop()

