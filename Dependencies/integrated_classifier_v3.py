"""
Integrated Classifier v3: Hybrid + LoRA with Expanded SLM Triggers

Key improvements:
1. SLM consulted for C1 grey area (tilt 20-35°) 
2. SLM consulted when C8 predicted but tilt elevated (maybe C1)
3. SLM consulted when C8 predicted but signal weak (maybe C3)
4. Improved prompt with explicit thresholds

This addresses:
- C1 cases missed due to tilt < 40° threshold
- C8 over-prediction when low RBs caused by coverage issues
"""

import os
os.environ['HF_HOME'] = '/export/scratch/abrar008/.cache/huggingface'

import re
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from seperate_values import parse_network_data, get_all_data_iterator
from generate_slm_prompts import get_pci_sequence, detect_pingpong_pattern, format_pci_sequence_string
from rule_based_classifier import extract_features
from xgboost_tool import XGBoostTool


class IntegratedClassifierV3:
    """Integrated classifier v3 with expanded SLM triggers."""
    
    TILT_THRESHOLD = 40  # Strict threshold for rules
    TILT_GREY_ZONE = (20, 35)  # Grey zone for SLM consultation
    RBS_THRESHOLD = 170
    SIGNAL_THRESHOLD = -95  # dBm - below this is "weak"
    
    def __init__(self, 
                 lora_model_path="lora_output_v3/final",
                 base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 xgboost_model_path="xgboost_model.pkl",
                 load_lora=True):
        
        self.xgboost_tool = XGBoostTool(xgboost_model_path)
        if self.xgboost_tool.model is None:
            self.xgboost_tool.train_model()
        
        self.model = None
        self.tokenizer = None
        self.load_lora = load_lora
        self.lora_model_path = lora_model_path
        self.base_model_name = base_model_name
        
        if load_lora and os.path.exists(lora_model_path):
            self._load_lora_model()
        elif load_lora:
            print(f"LoRA model not found at {lora_model_path}")
            self.load_lora = False
    
    def _load_lora_model(self):
        """Load the LoRA fine-tuned model."""
        print(f"Loading LoRA model from {self.lora_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lora_model_path,
            trust_remote_code=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
        self.model.eval()
        
        print(f"LoRA model loaded on {self.model.device}")
    
    def _get_signal_quality(self, df_up):
        """Extract mean signal quality from user plane data."""
        signal_col = '5G KPI PCell RF Serving SS-RSRP [dBm]'
        if signal_col in df_up.columns:
            values = pd.to_numeric(df_up[signal_col], errors='coerce')
            mean_signal = values.mean()
            if not pd.isna(mean_signal):
                return mean_signal
        return -90
    
    def classify(self, df_up, df_ep):
        """Classify with hybrid + expanded SLM triggers."""
        # Extract features
        features = extract_features(df_up, df_ep)
        pci_sequence = get_pci_sequence(df_up)
        is_pingpong = detect_pingpong_pattern(pci_sequence)
        mean_signal = self._get_signal_quality(df_up)
        
        xgb_result = self.xgboost_tool(features)
        
        # Phase 1: Apply strict hybrid rules
        hybrid_pred = self._apply_hybrid_rules(features, is_pingpong, xgb_result)
        method = "hybrid"
        slm_reason = None
        
        # Phase 2: Check if SLM should be consulted
        if self.load_lora:
            should_consult, reason = self._should_consult_slm(features, hybrid_pred, mean_signal)
            
            if should_consult:
                slm_pred = self._slm_classify(features, pci_sequence, is_pingpong, 
                                               hybrid_pred, mean_signal)
                if slm_pred != hybrid_pred:
                    method = f"slm_override_{reason}"
                    hybrid_pred = slm_pred
                    slm_reason = reason
        
        return {
            "prediction": hybrid_pred,
            "method": method,
            "slm_reason": slm_reason,
            "features": features,
            "xgboost_prediction": xgb_result["prediction"],
            "xgboost_confidence": xgb_result["confidence"],
            "is_pingpong": is_pingpong,
            "mean_signal": mean_signal
        }
    
    def _apply_hybrid_rules(self, features, is_pingpong, xgb_result):
        """Apply strict hybrid rules (tilt=40 threshold)."""
        # Physics layer
        if features['max_speed'] > 40:
            return "C7"
        if features['max_dist'] > 1.0:
            return "C2"
        
        # Interference layer
        if features['non_col_strong']:
            return "C4"
        if features['max_tilt'] > self.TILT_THRESHOLD:
            return "C1"
        if features['collision']:
            return "C6"
        
        # Optimization layer
        if is_pingpong:
            return "C5"
        if features['mean_rbs'] < self.RBS_THRESHOLD:
            return "C8"
        
        # XGBoost decision
        xgb_pred = xgb_result['prediction']
        if xgb_result['confidence'] > 0.6:
            if xgb_pred == "C5" and not is_pingpong:
                return "C3"
            return xgb_pred
        
        return "C3"
    
    def _should_consult_slm(self, features, hybrid_pred, mean_signal):
        """
        Determine if SLM should be consulted with reason.
        
        Returns: (should_consult: bool, reason: str)
        """
        tilt = features['max_tilt']
        
        # Case 1: C3 with elevated tilt (20-35°) → check for C1
        if hybrid_pred == "C3" and self.TILT_GREY_ZONE[0] <= tilt <= self.TILT_GREY_ZONE[1]:
            return True, "c3_tilt_grey"
        
        # Case 2: C8 with elevated tilt → check for C1 (tilt causes low RBs)
        if hybrid_pred == "C8" and tilt >= self.TILT_GREY_ZONE[0]:
            return True, "c8_tilt_elevated"
        
        # Case 3: C8 with weak signal → check for C3 (coverage causes low RBs)
        if hybrid_pred == "C8" and mean_signal < self.SIGNAL_THRESHOLD:
            return True, "c8_weak_signal"
        
        return False, None
    
    def _slm_classify(self, features, pci_sequence, is_pingpong, hybrid_pred, mean_signal):
        """Use SLM model with improved prompt including thresholds."""
        
        pci_str = format_pci_sequence_string(pci_sequence)
        
        # Improved prompt with explicit thresholds
        prompt = f"""5G Network Fault Diagnosis

MEASUREMENTS:
- Antenna Tilt: {features['max_tilt']:.1f}°
- Vehicle Speed: {features['max_speed']:.1f} km/h
- Cell Distance: {features['max_dist']:.2f} km
- Resource Blocks (RBs): {features['mean_rbs']:.0f}
- Signal Strength (SS-RSRP): {mean_signal:.1f} dBm
- PCI Collision: {features['collision']}
- Non-Colocated Interference: {features['non_col_strong']}
- Serving Cell Transitions: {pci_str}
- Ping-Pong Pattern: {is_pingpong}

CLASSIFICATION RULES (in priority order):
• C7: Speed > 40 km/h
• C2: Distance > 1.0 km (overshooting)
• C4: Strong non-colocated interference
• C6: PCI mod 30 collision
• C1: Tilt > 40° (strict) OR tilt 20-35° with coverage symptoms
• C5: A→B→A cell transition pattern (ping-pong)
• C8: RBs < 170 ONLY if good signal and no tilt issues
• C3: Coverage drift (default when none apply)

KEY INSIGHT: Low RBs can be CAUSED by:
- C1 (high tilt → weak coverage → fewer RBs)
- C3 (coverage drift → weak signal → fewer RBs)
- C8 (true congestion → many users → fewer RBs per user)

Hybrid classifier predicts: {hybrid_pred}
Verify this prediction. Is it correct?

Analyze step-by-step following the rules above."""
        
        # Run inference
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse prediction
        match = re.search(r'Final Answer:\s*(C[1-8])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        matches = re.findall(r'\b(C[1-8])\b', response)
        if matches:
            return matches[-1].upper()
        
        return hybrid_pred


def evaluate_classifier(classifier, mode='train', limit=None, verbose=False):
    """Evaluate the integrated classifier v3."""
    predictions = []
    labels = []
    methods = {"hybrid": 0, "slm_override_c3_tilt_grey": 0, 
               "slm_override_c8_tilt_elevated": 0, "slm_override_c8_weak_signal": 0}
    
    if mode == 'train':
        print("Evaluating on training data...")
        sample_count = 0
        
        for index, raw_row, df_up, df_ep in get_all_data_iterator():
            truth_match = re.search(r'(C[1-8])', str(raw_row['answer']))
            if not truth_match:
                continue
            
            true_label = truth_match.group(1)
            result = classifier.classify(df_up, df_ep)
            
            predictions.append(result["prediction"])
            labels.append(true_label)
            
            method = result["method"]
            if method in methods:
                methods[method] += 1
            else:
                methods["hybrid"] += 1
            
            if verbose:
                status = "✓" if result["prediction"] == true_label else "✗"
                print(f"{status} {index}: True={true_label}, Pred={result['prediction']}, Method={result['method']}")
            
            sample_count += 1
            if limit and sample_count >= limit:
                break
                
    elif mode == 'test':
        print("Evaluating on test data (phase_1_test.csv)...")
        test_df = pd.read_csv("phase_1_test.csv")
        truth_df = pd.read_csv("phase_1_test_truth.csv")
        
        truth_map = {}
        for _, row in truth_df.iterrows():
            full_id = str(row['ID'])
            base_id = full_id.rsplit('_', 1)[0]
            if base_id not in truth_map:
                truth_map[base_id] = row.iloc[1]
        
        sample_count = 0
        for idx, row in test_df.iterrows():
            row_id = str(row['ID'])
            if row_id not in truth_map:
                continue
            
            true_label = truth_map[row_id]
            raw_text = row['question']
            df_up, df_ep = parse_network_data(raw_text)
            
            result = classifier.classify(df_up, df_ep)
            
            predictions.append(result["prediction"])
            labels.append(true_label)
            
            method = result["method"]
            if method in methods:
                methods[method] += 1
            else:
                methods["hybrid"] += 1
            
            if verbose:
                status = "✓" if result["prediction"] == true_label else "✗"
                print(f"{status} {row_id}: True={true_label}, Pred={result['prediction']}")
            
            sample_count += 1
            if limit and sample_count >= limit:
                break
            
            if sample_count % 100 == 0:
                print(f"  Processed {sample_count} samples...")
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    print(f"\n{'='*60}")
    print(f"INTEGRATED CLASSIFIER V3 RESULTS ({mode.upper()} SET)")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(labels)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    print(f"\nMethod Usage:")
    total = sum(methods.values())
    for method, count in methods.items():
        if count > 0:
            print(f"  {method}: {count} ({100*count/total:.1f}%)")
    
    print(f"\nClassification Report:")
    print(classification_report(labels, predictions))
    
    # Confusion matrix
    print("Confusion Matrix:")
    classes = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=classes)
    print(f"         Predicted")
    print(f"         {' '.join(f'{c:>4}' for c in classes)}")
    for i, c in enumerate(classes):
        print(f"True {c}: {' '.join(f'{cm[i,j]:>4}' for j in range(len(classes)))}")
    
    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "labels": labels,
        "methods": methods
    }


def main():
    parser = argparse.ArgumentParser(description="Integrated Classifier v3")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora-path", default="lora_output_v3/final")
    
    args = parser.parse_args()
    
    classifier = IntegratedClassifierV3(
        lora_model_path=args.lora_path,
        load_lora=not args.no_lora
    )
    
    results = evaluate_classifier(
        classifier,
        mode=args.mode,
        limit=args.limit,
        verbose=args.verbose
    )
    
    return results


if __name__ == "__main__":
    main()
