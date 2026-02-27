"""
MOE (Mixture of Experts) Submission Generator

Uses:
- v5d model for Phase 1 questions (C1-C8 format)
- Phase 2 model for Phase 2 questions (jumbled options A-I, 1-8, etc.)
"""

import os
os.environ['HF_HOME'] = '/export/scratch/abrar008/.cache/huggingface'

import pandas as pd
import re
import torch
from tqdm import tqdm
from seperate_values import parse_network_data
from integrated_classifier_v3 import IntegratedClassifierV3
from rule_based_classifier import extract_features
from xgboost_tool import XGBoostTool
from generate_slm_prompts import get_pci_sequence, detect_pingpong_pattern

# Class descriptions for expert hints
CLASS_DESCRIPTIONS = {
    'C1': 'Antenna Tilt / Weak Coverage due to Tilt',
    'C2': 'Over-shooting / Coverage Distance too large',
    'C3': 'Coverage Drift / Missing Neighbor / Better Cell exists',
    'C4': 'Strong Interference / Overlap / Jamming',
    'C5': 'Ping-Pong Handover / Frequent Handovers',
    'C6': 'PCI Collision / Mod 30 Collision',
    'C7': 'High Speed / Velocity > 40km/h',
    'C8': 'Congestion / Low RBs / Capacity Insufficient'
}


def format_answer(prediction):
    """Format the prediction with double backslash for submission."""
    return f"Based on the analysis of 5G network measurements, the root cause is: \\\\boxed{{{prediction}}}"


def create_v5d_prompt(features, xgb_pred, xgb_conf, is_pingpong):
    """Create Phase 1 style prompt for v5d model."""
    tilt = features['max_tilt']
    speed = features['max_speed']
    dist = features['max_dist']
    rbs = features['mean_rbs']
    collision = features['collision']
    interference = features['non_col_strong']
    
    tilt_flag = " [GREY]" if 20 <= tilt <= 35 else " [HIGH]" if tilt > 35 else ""
    speed_flag = " [HIGH]" if speed > 40 else ""
    dist_flag = " [FAR]" if dist > 1.0 else ""
    rbs_flag = " [LOW]" if rbs < 170 else ""
    pingpong_flag = " [TRIGGER]" if is_pingpong else ""
    
    prompt = f"""5G Fault Classification

Measurements:
- Tilt: {tilt:.0f}°{tilt_flag}
- Speed: {speed:.0f} km/h{speed_flag}
- Distance: {dist:.2f} km{dist_flag}
- RBs: {rbs:.0f}{rbs_flag}
- Collision: {'Yes' if collision else 'No'}
- Interference: {'Yes' if interference else 'No'}
- Ping-Pong: {'Yes' if is_pingpong else 'No'}{pingpong_flag}

XGBoost: {xgb_pred} ({xgb_conf:.0%})

Options: C1=Tilt, C2=Overshoot, C3=Drift, C4=Jamming, C5=Handover, C6=Collision, C7=Speed, C8=Congestion
Classify and output \\boxed{{answer}}."""
    
    return prompt


def create_phase2_prompt(raw_question, hint):
    """Create Phase 2 style prompt with expert hint."""
    prompt = f"{raw_question}{hint}"
    return prompt


def create_expert_hint(features, xgb_pred, xgb_conf, is_pingpong):
    """Create expert analysis hint for Phase 2."""
    triggers = []
    
    if features['max_speed'] > 40:
        triggers.append(f"High speed ({features['max_speed']:.0f}km/h)")
    if features['max_tilt'] > 40:
        triggers.append(f"High tilt ({features['max_tilt']:.0f}°)")
    if features['max_dist'] > 1.0:
        triggers.append(f"Overshoot (dist {features['max_dist']:.2f}km)")
    if features['mean_rbs'] < 170:
        triggers.append(f"Low RBs ({features['mean_rbs']:.0f})")
    if features['collision']:
        triggers.append("PCI collision")
    if is_pingpong:
        triggers.append("Ping-pong handover")
    
    desc = CLASS_DESCRIPTIONS.get(xgb_pred, 'Unknown')
    trig_text = f" ({', '.join(triggers)})" if triggers else ""
    
    hint = f"\n\n[Expert Analysis]\nNetwork analysis suggests: {desc}{trig_text}.\nSelect the option that best matches this cause."
    return hint


def generate_slm_response(classifier, prompt, force_c_prefix=False, xgb_fallback="C3"):
    """Generate response from SLM with improved label extraction."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = classifier.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = classifier.tokenizer([text], return_tensors="pt").to(classifier.model.device)
    
    with torch.no_grad():
        outputs = classifier.model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.1,
            do_sample=True,
            pad_token_id=classifier.tokenizer.eos_token_id
        )
    response = classifier.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Priority 1: Look for C1-C8 in boxed{} format
    c_matches = re.findall(r'boxed\{(C[1-8])\}', response)
    if c_matches:
        return c_matches[-1]
        
    # Priority 2: Look for Letter+Digit (I3, A1, B5, etc.) match in text
    # (Matches A1-Z9 patterns which are strong indicators of specific labels)
    # Get only the generated part (after prompt)
    if 'assistant' in response:
        gen_part = response.split('assistant')[-1]
    else:
        gen_part = response[-300:]
        
    ld_matches = re.findall(r'\b([A-Z][0-9])\b', gen_part)
    if ld_matches:
        # Verify it's not a false positive like "V3" in "IntegratedClassifierV3"
        # Only accept if length is 2 or 3
        valid_ld = [m for m in ld_matches if len(m) <= 3]
        if valid_ld:
            return valid_ld[-1]
    
    # Priority 3: Look for any valid boxed match (not 'answer')
    all_matches = re.findall(r'boxed\{([^\}]+)\}', response)
    for match in reversed(all_matches):
        if match.lower() != 'answer' and len(match) <= 3:
            return _format_label(match, force_c_prefix)
    
    # Priority 4: Look for C1-C8 specific match in text (if prediction was force_c_prefix)
    if force_c_prefix:
        text_c_matches = re.findall(r'\b(C[1-8])\b', gen_part)
        if text_c_matches:
            return text_c_matches[-1]
    
    # Priority 5: Look for digit labels (1-8) for Phase 2
    if not force_c_prefix:
        # Check text for single digit answers
        text_digit_matches = re.findall(r'\b([1-8])\b', gen_part)
        if text_digit_matches:
            return text_digit_matches[-1]
    
    # Fallback to XGBoost prediction
    return xgb_fallback if force_c_prefix else "1"

def _format_label(label, force_c_prefix=False):
    """Format the extracted label."""
    label = re.sub(r'[^A-Za-z0-9]', '', label)
    if not label:
        return "C3" if force_c_prefix else "1"
    
    if force_c_prefix:
        # ... (keep existing logic)
        if label.isdigit():
            return f"C{label}"
        if len(label) == 1 and label.isalpha():
            num = ord(label.upper()) - ord('A') + 1
            return f"C{num}"
    
    return label.upper() # Return exact label for Phase 2 (I3, A1, 1, etc.)


def process_phase1(test_file, v5d_classifier, xgb_tool):
    """Process Phase 1 with v5d model."""
    # ... (no changes)
    print(f"\nProcessing {test_file} with v5d model...")
    df = pd.read_csv(test_file)
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Phase 1 v5d"):
        row_id = str(row['ID'])
        raw_text = row['question']
        
        # Parse network data
        try:
            df_up, df_ep = parse_network_data(raw_text)
            if df_up.empty:
                pred = "C3"
            else:
                features = extract_features(df_up, df_ep)
                pci_seq = get_pci_sequence(df_up)
                is_pingpong = detect_pingpong_pattern(pci_seq)
                xgb_result = xgb_tool(features)
                
                prompt = create_v5d_prompt(features, xgb_result['prediction'], 
                                           xgb_result['confidence'], is_pingpong)
                pred = generate_slm_response(v5d_classifier, prompt, force_c_prefix=True, 
                                             xgb_fallback=xgb_result['prediction'])
        except:
            pred = "C3"
        
        # 4 temperature variations (all same for v5d)
        for i in range(1, 5):
            results.append({
                'ID': f"{row_id}_{i}",
                'Qwen3-32B': 'placeholder',
                'Qwen2.5-7B-Instruct': 'placeholder',
                'Qwen2.5-1.5B-Instruct': format_answer(pred)
            })
    
    return results


def process_phase2(test_file, phase2_classifier, xgb_tool):
    # ... (no changes)
    print(f"\nProcessing {test_file} with Phase 2 model...")
    df = pd.read_csv(test_file)
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Phase 2 Adaptive"):
        row_id = str(row['ID'])
        raw_text = row['question']
        
        # Try to add expert hint
        hint = ""
        try:
            df_up, df_ep = parse_network_data(raw_text)
            if not df_up.empty:
                features = extract_features(df_up, df_ep)
                pci_seq = get_pci_sequence(df_up)
                is_pingpong = detect_pingpong_pattern(pci_seq)
                xgb_result = xgb_tool(features)
                hint = create_expert_hint(features, xgb_result['prediction'], 
                                          xgb_result['confidence'], is_pingpong)
        except:
            pass
        
        prompt = create_phase2_prompt(raw_text, hint)
        pred = generate_slm_response(phase2_classifier, prompt, force_c_prefix=False)
        
        for i in range(1, 5):
            results.append({
                'ID': f"{row_id}_{i}",
                'Qwen3-32B': 'placeholder',
                'Qwen2.5-7B-Instruct': 'placeholder',
                'Qwen2.5-1.5B-Instruct': format_answer(pred)
            })
    
    return results


def generate_moe_submission():
    """Generate MOE submission using v5d for Phase 1 and Phase 2 model for Phase 2."""
    print("="*60)
    print("MOE SUBMISSION GENERATOR")
    print("="*60)
    
    # Load v5d model for Phase 1
    print("\nLoading v5d model for Phase 1...")
    v5d_classifier = IntegratedClassifierV3(lora_model_path="lora_output_v5d/final")
    xgb_tool = XGBoostTool("xgboost_model.pkl")
    
    # Process Phase 1
    phase1_results = process_phase1("phase_1_test.csv", v5d_classifier, xgb_tool)
    
    # Unload v5d model
    del v5d_classifier
    torch.cuda.empty_cache()
    
    # Load Phase 2 model (UPDATE TO V3)
    print("\nLoading Phase 2 model (v3)...")
    phase2_classifier = IntegratedClassifierV3(lora_model_path="lora_output_phase2_v3/final")
    
    # Process Phase 2
    phase2_results = process_phase2("phase_2_test.csv", phase2_classifier, xgb_tool)
    
    # Combine results
    all_results = phase1_results + phase2_results
    df = pd.DataFrame(all_results)
    
    # Save
    output_file = "submission_moe.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("MOE SUBMISSION COMPLETE")
    print("="*60)
    print(f"Output: {output_file}")
    print(f"Total rows: {len(df)} (Phase 1: {len(phase1_results)}, Phase 2: {len(phase2_results)})")
    
    # Show samples
    print("\n=== Sample Phase 1 rows ===")
    print(df.head(3).to_string())
    
    print("\n=== Sample Phase 2 rows ===")
    print(df.iloc[len(phase1_results):len(phase1_results)+3].to_string())


if __name__ == "__main__":
    generate_moe_submission()
