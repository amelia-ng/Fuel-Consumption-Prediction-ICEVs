Accurate prediction of fuel consumption in internal combustion engine vehicles (ICEVs) is a prerequisite for effective fleet management, emissions monitoring, and powertrain diagnostics. This project benchmarks a comprehensive suite of machine learning models, spanning tree-based ensemble methods and deep sequence models on a large-scale real-world telematics dataset comprising over 12.4 million observations. Models evaluated include Ordinary Least Squares (OLS) variants, Random Forest, Gradient Boosting, XGBoost, LightGBM (both with and without Optuna hyperparameter tuning), and Long Short-Term Memory (LSTM) networks. For log-transform model variants, predictions are back-transformed to the original L/hr scale before evaluation, ensuring all reported metrics are directly interpretable in physical units. 

Results demonstrate that gradient-boosted tree models with log-transformed targets and Optuna hyperparameter search achieve the best generalization, with the tuned LightGBM achieving the best test set performance (MAE: 0.3070 L/hr; RMSE: 0.5343 L/hr; R-squared: 0.6419; MAPE: 13.87%). The LSTM, while being a natural candidate for sequential driving data, underperforms tree-based methods across all metrics. SHAP analysis confirms that vehicle speed, rolling-speed statistics, engine displacement, and acceleration dominate the model's output, consistent with the physics of fuel consumption. These findings provide a practical guide for practitioners selecting models for real-world ICEV fuel prediction.

Key achievements:
- Approach the data leakage issue
- Final model reached 13.87% MAPE and 0.3 MAE (L/hr) on **real-life** dataset (Feature engineering that brings MAPE from ~55% to 13.87%)
- SHAP for feature importance analysis


**Article summarizing result: https://medium.com/@ameliablog/benchmarking-machine-learning-approach-for-real-world-fuel-consumption-prediction-in-icevs-f657408e888b**

