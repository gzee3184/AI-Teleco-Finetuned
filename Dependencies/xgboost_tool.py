"""
XGBoost Tool Module - Provides XGBoost as a callable tool for SLM.
"""

import pickle
import os
import pandas as pd
import numpy as np
import re
try:
    import xgboost as xgb
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as xgb

from sklearn.preprocessing import LabelEncoder
from seperate_values import get_all_data_iterator, parse_network_data
from rule_based_classifier import extract_features


class XGBoostTool:
    """Tool that SLM can call to get XGBoost's classification."""
    
    def __init__(self, model_path="xgboost_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model not found at {model_path}. Call train_model() first.")
    
    def train_model(self):
        """Train XGBoost on the training data and save."""
        print("Training XGBoost model...")
        
        data = []
        labels = []
        
        for index, raw_row, df_up, df_ep in get_all_data_iterator():
            truth_match = re.search(r'(C[1-8])', str(raw_row['answer']))
            if not truth_match:
                continue
            
            feats = extract_features(df_up, df_ep)
            data.append(feats)
            labels.append(truth_match.group(1))
        
        X_train = pd.DataFrame(data)
        self.feature_names = list(X_train.columns)
        
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(labels)
        
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(self.label_encoder.classes_),
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        self.save_model()
        print(f"Model trained and saved to {self.model_path}")
        return self
    
    def save_model(self):
        """Save model and encoder to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, f)
    
    def load_model(self):
        """Load model and encoder from disk."""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.feature_names = data['feature_names']
        print(f"Model loaded from {self.model_path}")
    
    def __call__(self, features: dict) -> dict:
        """
        Classify using XGBoost.
        
        Args:
            features: dict with max_speed, max_dist, mean_rbs, ho_count, etc.
            
        Returns:
            dict with prediction, confidence, and all class probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train_model() or load a saved model.")
        
        # Ensure features are in correct order
        feature_df = pd.DataFrame([features])[self.feature_names]
        
        # Get prediction
        pred_encoded = self.model.predict(feature_df)[0]
        prediction = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get probabilities
        proba = self.model.predict_proba(feature_df)[0]
        confidence = float(max(proba))
        
        # Map probabilities to class names
        all_probs = {}
        for i, cls in enumerate(self.label_encoder.classes_):
            all_probs[cls] = float(proba[i])
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "all_probabilities": all_probs
        }
    
    def classify_from_dataframes(self, df_up, df_ep):
        """Convenience method to classify directly from parsed dataframes."""
        features = extract_features(df_up, df_ep)
        return self(features)


# Tool schema for SLM function calling
XGBOOST_TOOL_SCHEMA = {
    "name": "classify_with_xgboost",
    "description": "Call XGBoost classifier on scalar features. Returns prediction and confidence. Use this for initial classification, then verify with temporal analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "features": {
                "type": "object",
                "description": "Scalar features: max_speed, max_dist, mean_rbs, ho_count, max_tilt, collision, non_col_strong",
                "properties": {
                    "max_speed": {"type": "number"},
                    "max_dist": {"type": "number"},
                    "mean_rbs": {"type": "number"},
                    "ho_count": {"type": "integer"},
                    "max_tilt": {"type": "number"},
                    "collision": {"type": "boolean"},
                    "non_col_strong": {"type": "boolean"}
                }
            }
        },
        "required": ["features"]
    }
}


if __name__ == "__main__":
    # Train and save model
    tool = XGBoostTool("xgboost_model.pkl")
    tool.train_model()
    
    # Test with a sample
    print("\nTesting tool with sample features...")
    sample_features = {
        'max_tilt': 10,
        'max_dist': 0.5,
        'non_col_strong': False,
        'ho_count': 2,
        'collision': False,
        'max_speed': 25,
        'mean_rbs': 200
    }
    result = tool(sample_features)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"All probabilities: {result['all_probabilities']}")
