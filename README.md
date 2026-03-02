# BakeOpt
Tools to analyze data of pizza baking experiments and predict input parameters (temperature and time) to obtain the desired browning index and humidity loss.

There are two scripts:
- model_fitter_analyzer.py allows to analyze and fit data with linear/quadratic/cubic models, which can be exported. Data must be imported as .csv files organized like the one in this repository
- model_inverter.py allows to import the fitted model, and invert the relationship between inputs (oven temperatures and time) and outputs (Browning indices and humidity loss).

Developed with Python 3.12 and libraries as detailed in requirements.txt. Minimal requirements are listed in minimal_requirements.txt.
