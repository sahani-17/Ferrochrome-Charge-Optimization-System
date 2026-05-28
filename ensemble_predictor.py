"""
Ensemble Predictor - Ferrochrome Optimization
Phase 4: Ensemble Methods

Loads and uses ensemble models for prediction.
Combines predictions from multiple models using weighted averaging.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import pickle
from pathlib import Path

class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models for Cr recovery prediction.
    """
    
    def __init__(self, model_dir='models/ensemble'):
        """
        Initialize ensemble predictor.
        
        Args:
            model_dir: Directory containing ensemble models
        """
        self.model_dir = model_dir
        self.models = {}
        self.weights = {}
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all ensemble models and metadata."""
        print(f"Loading ensemble models from {self.model_dir}...")
        
        # Load metadata
        with open(f'{self.model_dir}/ensemble_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load weights
        self.weights = self.metadata['weights']
        
        # Load scaler
        with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        with open(f'{self.model_dir}/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Load LightGBM models
        lgbm_models = []
        for seed in self.metadata['models']['lgbm_seeds']:
            model = lgb.Booster(model_file=f'{self.model_dir}/lgbm_cr_seed_{seed}.txt')
            lgbm_models.append(model)
        self.models['lgbm_ensemble'] = lgbm_models
        
        # Load XGBoost model
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(f'{self.model_dir}/xgb_cr_recovery.json')
        self.models['xgboost'] = xgb_model
        
        # Load CatBoost model
        cat_model = CatBoostRegressor()
        cat_model.load_model(f'{self.model_dir}/catboost_cr_recovery.cbm')
        self.models['catboost'] = cat_model
        
        # Load Random Forest model
        with open(f'{self.model_dir}/rf_cr_recovery.pkl', 'rb') as f:
            self.models['random_forest'] = pickle.load(f)
        
        print(f"  Loaded {len(self.metadata['models']['lgbm_seeds'])} LightGBM models")
        print(f"  Loaded XGBoost model")
        print(f"  Loaded CatBoost model")
        print(f"  Loaded Random Forest model")
        print(f"  Ensemble ready!")
    
    def predict(self, X):
        """
        Predict Cr recovery using ensemble.
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
        
        Returns:
            Predictions (numpy array)
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        lgbm_preds = np.array([model.predict(X_scaled) for model in self.models['lgbm_ensemble']])
        lgbm_ensemble_pred = np.mean(lgbm_preds, axis=0)
        
        xgb_pred = self.models['xgboost'].predict(X_scaled)
        cat_pred = self.models['catboost'].predict(X_scaled)
        rf_pred = self.models['random_forest'].predict(X_scaled)
        
        # Weighted ensemble prediction
        ensemble_pred = (
            self.weights['lgbm_ensemble'] * lgbm_ensemble_pred +
            self.weights['xgboost'] * xgb_pred +
            self.weights['catboost'] * cat_pred +
            self.weights['random_forest'] * rf_pred
        )
        
        return ensemble_pred
    
    def predict_with_breakdown(self, X):
        """
        Predict with breakdown of individual model contributions.
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
        
        Returns:
            Dictionary with ensemble prediction and individual model predictions
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        lgbm_preds = np.array([model.predict(X_scaled) for model in self.models['lgbm_ensemble']])
        lgbm_ensemble_pred = np.mean(lgbm_preds, axis=0)
        
        xgb_pred = self.models['xgboost'].predict(X_scaled)
        cat_pred = self.models['catboost'].predict(X_scaled)
        rf_pred = self.models['random_forest'].predict(X_scaled)
        
        # Weighted ensemble prediction
        ensemble_pred = (
            self.weights['lgbm_ensemble'] * lgbm_ensemble_pred +
            self.weights['xgboost'] * xgb_pred +
            self.weights['catboost'] * cat_pred +
            self.weights['random_forest'] * rf_pred
        )
        
        return {
            'ensemble': ensemble_pred,
            'lgbm_ensemble': lgbm_ensemble_pred,
            'xgboost': xgb_pred,
            'catboost': cat_pred,
            'random_forest': rf_pred,
            'weights': self.weights
        }
    
    def get_performance_metrics(self):
        """
        Get ensemble performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.metadata['metrics']
    
    def get_model_weights(self):
        """
        Get ensemble model weights.
        
        Returns:
            Dictionary with model weights
        """
        return self.weights


# Example usage
if __name__ == '__main__':
    print("="*80)
    print("ENSEMBLE PREDICTOR TEST")
    print("="*80)
    
    # Load ensemble
    predictor = EnsemblePredictor()
    
    # Display performance
    print(f"\nEnsemble Performance Metrics:")
    metrics = predictor.get_performance_metrics()
    for metric, value in metrics.items():
        print(f"  {metric.upper():10s}: {value:.4f}")
    
    # Display weights
    print(f"\nModel Weights:")
    weights = predictor.get_model_weights()
    for model, weight in weights.items():
        print(f"  {model:20s}: {weight:.4f}")
    
    # Test prediction
    print(f"\nTesting prediction on sample data...")
    df = pd.read_csv('FINAL_DATASET.csv')
    df_clean = df.dropna(subset=['Ore_qty', 'Flux_qty', 'Reductant_qty', 'Additive_qty',
                                  'Cr_recovery_pct', 'SP_Power'])
    
    # Get first 5 samples
    X_sample = df_clean[predictor.feature_names].head(5)
    y_actual = df_clean['Cr_recovery_pct'].head(5).values
    
    # Predict
    predictions = predictor.predict(X_sample)
    
    print(f"\nSample Predictions:")
    print(f"  {'Actual':>10s}  {'Predicted':>10s}  {'Error':>10s}")
    for actual, pred in zip(y_actual, predictions):
        error = pred - actual
        print(f"  {actual:10.2f}  {pred:10.2f}  {error:+10.2f}")
    
    print(f"\n{'='*80}")
    print("Ensemble predictor ready for use!")
    print("="*80)
