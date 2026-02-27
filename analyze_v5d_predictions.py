import pandas as pd
import re
from collections import Counter

def extract_answer_robust(text):
    """
    Extract C[1-8] answer from text using multiple patterns.
    Looks for: \boxed{CX}, "answer is CX", "answer is: CX", etc.
    """
    if pd.isna(text):
        return None
    
    text = str(text)
    
    # Try multiple patterns in order of preference
    patterns = [
        r'\\boxed\{(C[1-8])\}',  # \boxed{CX}
        r'answer is:?\s+(C[1-8])',  # "answer is: CX" or "answer is CX"
        r'Output:\s+(C[1-8])',  # "Output: CX"
        r'\b(C[1-8])\b(?!\d)',  # Any standalone C[1-8] (last resort)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None

def main():
    print("="*70)
    print(" v5d Phase 1 Test Failure Analysis (IMPROVED EXTRACTION)")
    print("="*70)
    
    # Load predictions
    pred = pd.read_csv('v5d_phase1_predictions.csv')
    print(f"\nPredictions loaded: {len(pred)} base questions")
    
    # Re-extract with improved regex
    pred['predicted_improved'] = pred['full_output'].apply(extract_answer_robust)
    
    print(f"Original extraction: {pred['predicted'].notna().sum()} successful")
    print(f"Improved extraction: {pred['predicted_improved'].notna().sum()} successful")
    print(f"Improvement: +{pred['predicted_improved'].notna().sum() - pred['predicted'].notna().sum()} predictions recovered")
    
    # Load truth
    truth = pd.read_csv('phase_1_test_truth.csv')
    truth['base_id'] = truth['ID'].str.rsplit('_', n=1).str[0]
    truth_grouped = truth.groupby('base_id')['Qwen2.5-1.5B-Instruct'].apply(set).to_dict()
    
    # Evaluate
    results = []
    for _, row in pred.iterrows():
        base_id = row['ID']
        predicted = row['predicted_improved']
        
        if base_id not in truth_grouped:
            continue
        
        truth_options = truth_grouped[base_id]
        correct = predicted in truth_options if pd.notna(predicted) else False
        actual_truth = list(truth_options)[0] if len(truth_options) == 1 else str(truth_options)
        
        results.append({
            'base_id': base_id,
            'predicted': predicted,
            'truth': actual_truth,
            'correct': correct,
            'parsed': pd.notna(predicted)
        })
    
    df = pd.DataFrame(results)
    
    # Statistics
    total = len(df)
    parsed = df['parsed'].sum()
    correct = df['correct'].sum()
    
    print(f"\n{'='*70}")
    print(f" RESULTS")
    print(f"{'='*70}")
    print(f"Total base questions: {total}")
    print(f"Successfully parsed: {parsed} ({100*parsed/total:.1f}%)")
    print(f"Parse failures: {total-parsed} ({100*(total-parsed)/total:.1f}%)")
    print(f"\nOf parsed predictions:")
    print(f"  Correct: {correct}/{parsed} ({100*correct/parsed:.1f}%)")
    print(f"  Incorrect: {parsed-correct}/{parsed} ({100*(parsed-correct)/parsed:.1f}%)")
    print(f"\nOverall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Per-class accuracy
    print(f"\n{'='*70}")
    print(f" Per-Class Accuracy")
    print(f"{'='*70}")
    df_parsed = df[df['parsed']].copy()
    for cls in sorted(df_parsed['truth'].unique()):
        cls_mask = df_parsed['truth'] == cls
        cls_total = cls_mask.sum()
        cls_correct = (df_parsed['correct'] & cls_mask).sum()
        print(f"  {cls}: {cls_correct}/{cls_total} ({100*cls_correct/cls_total:.1f}%)")
    
    # Confusion matrix
    failures = df_parsed[~df_parsed['correct']].copy()
    if len(failures) > 0:
        print(f"\n{'='*70}")
        print(f" Top Failure Patterns (Truth -> Predicted)")
        print(f"{'='*70}")
        conf = failures.groupby(['truth', 'predicted']).size().sort_values(ascending=False)
        for (t, p), count in list(conf.items())[:20]:
            print(f"  {t} -> {p}: {count} failures")
    
    # Class distribution
    print(f"\n{'='*70}")
    print(f" Class Distribution")
    print(f"{'='*70}")
    pred_dist = Counter(df_parsed['predicted'])
    truth_dist = Counter(df_parsed['truth'])
    all_classes = sorted(set(list(pred_dist.keys()) + list(truth_dist.keys())))
    
    print(f"  {'Class':<8} {'Predicted':<12} {'Truth':<12} {'Diff':<8}")
    for cls in all_classes:
        p = pred_dist.get(cls, 0)
        t = truth_dist.get(cls, 0)
        print(f"  {cls:<8} {p:<12} {t:<12} {p-t:+d}")
    
    # Save
    df.to_csv('v5d_phase1_analysis_improved.csv', index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to v5d_phase1_analysis_improved.csv")
    
    # Show remaining parse failures
    remaining_fails = df[~df['parsed']].head(5)
    if len(remaining_fails) > 0:
        print(f"\n{'='*70}")
        print(f" Sample Remaining Parse Failures ({total-parsed} total)")
        print(f"{'='*70}")
        for _, row in remaining_fails.iterrows():
            full_pred = pred[pred['ID'] == row['base_id']]['full_output'].iloc[0]
            print(f"\nID: {row['base_id']}")
            print(f"Truth: {row['truth']}")
            print(f"Output: {full_pred[:200]}...")

if __name__ == "__main__":
    main()
