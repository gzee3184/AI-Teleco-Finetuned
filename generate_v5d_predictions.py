"""
Generate v5d predictions for train.csv and phase_1_test.csv
Outputs: v5d_train_predictions.csv and v5d_phase1_predictions.csv
"""
import os
os.environ['HF_HOME'] = '/export/scratch/abrar008/.cache/huggingface' # Change to HF path

import pandas as pd
import torch
from tqdm import tqdm
import re
from integrated_classifier_v3 import IntegratedClassifierV3
from xgboost_tool import XGBoostTool
from seperate_values import parse_network_data
from rule_based_classifier import extract_features
from generate_submissions_moe import create_v5d_prompt, get_pci_sequence, detect_pingpong_pattern

def extract_answer(text):
    """Extract CX from \\boxed{CX} format."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1)
    return None

def predict_sample(row, classifier, xgb_tool):
    """Generate v5d prediction for a single sample."""
    question = row['question']
    
    try:
        df_up, df_ep = parse_network_data(question)
        if df_up.empty:
            return None, "Empty network data"
        
        features = extract_features(df_up, df_ep)
        pci_seq = get_pci_sequence(df_up)
        is_pingpong = detect_pingpong_pattern(pci_seq)
        xgb_res = xgb_tool(features)
        
        # Create v5d prompt
        prompt = create_v5d_prompt(features, xgb_res['prediction'], xgb_res['confidence'], is_pingpong)
        
        # Generate
        messages = [{"role": "user", "content": prompt}]
        text = classifier.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = classifier.tokenizer([text], return_tensors="pt").to(classifier.model.device)
        
        with torch.no_grad():
            outputs = classifier.model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=True)
        
        response = classifier.tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen = response.split('assistant')[-1].strip()
        
        # Extract answer
        answer = extract_answer(gen)
        return answer, gen
    except Exception as e:
        return None, str(e)

def main():
    print("Loading v5d model...")
    classifier = IntegratedClassifierV3(lora_model_path="lora_output_v5d/final")
    xgb_tool = XGBoostTool("xgboost_model.pkl")
    
    # ================================================================
    # 1. Generate predictions for phase_1_test.csv
    # ================================================================
    print("\n=== Processing phase_1_test.csv ===")
    df_test = pd.read_csv("phase_1_test.csv")
    print(f"Total samples: {len(df_test)}")
    
    predictions_test = []
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Phase 1 Test"):
        answer, full_output = predict_sample(row, classifier, xgb_tool)
        predictions_test.append({
            'ID': row['ID'],
            'predicted': answer,
            'full_output': full_output
        })
    
    pred_df_test = pd.DataFrame(predictions_test)
    pred_df_test.to_csv('v5d_phase1_predictions.csv', index=False)
    print(f"Saved predictions to v5d_phase1_predictions.csv")
    print(f"  Successful: {pred_df_test['predicted'].notna().sum()}")
    print(f"  Failed: {pred_df_test['predicted'].isna().sum()}")
    
    # ================================================================
    # 2. Generate predictions for train.csv
    # ================================================================
    print("\n=== Processing train.csv ===")
    df_train = pd.read_csv("train.csv")
    print(f"Total samples: {len(df_train)}")
    
    predictions_train = []
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Train"):
        answer, full_output = predict_sample(row, classifier, xgb_tool)
        predictions_train.append({
            'ID': row['ID'],
            'predicted': answer,
            'full_output': full_output
        })
    
    pred_df_train = pd.DataFrame(predictions_train)
    pred_df_train.to_csv('v5d_train_predictions.csv', index=False)
    print(f"Saved predictions to v5d_train_predictions.csv")
    print(f"  Successful: {pred_df_train['predicted'].notna().sum()}")
    print(f"  Failed: {pred_df_train['predicted'].isna().sum()}")

if __name__ == "__main__":
    main()
