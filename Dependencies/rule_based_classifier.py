import os
import pandas as pd
import math
import re
from seperate_values import parse_network_data
import csv

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def run_classification():
    test_file = "phase_1_test.csv"
    output_file = "phase_1_predictions.csv"
    
    print(f"Reading {test_file}...")
    df_test = pd.read_csv(test_file)
    
    predictions = []
    
    print("Classifying samples...")
    for idx, row in df_test.iterrows():
        raw_text = row['question']
        row_id = row['ID']
        
        # Reuse existing parser
        df_up, df_ep = parse_network_data(raw_text)
        
        if df_up.empty:
            predictions.append({'ID': row_id, 'Predicted_Cause': 'C3'}) # Default
            continue

        # --- PREPROCESSING (Same as before) ---
        cols_to_numeric = [
            '5G KPI PCell RF Serving SS-RSRP [dBm]',
            '5G KPI PCell Layer1 DL RB Num (Including 0)',
            '5G KPI PCell RF Serving PCI',
            'GPS Speed (km/h)',
            'Latitude', 'Longitude'
        ]
        for c in cols_to_numeric:
            if c in df_up.columns:
                df_up[c] = pd.to_numeric(df_up[c], errors='coerce')

        # Build Maps
        pci_ep_map = {}
        pci_gnodeb_map = {}
        if not df_ep.empty:
            for _, ep_r in df_ep.iterrows():
                try:
                    p = int(ep_r['PCI'])
                    mech = float(ep_r['Mechanical Downtilt']) if 'Mechanical Downtilt' in ep_r else 0
                    dig = float(ep_r['Digital Tilt']) if 'Digital Tilt' in ep_r else 0
                    if dig == 255: dig = 6.0
                    
                    pci_ep_map[p] = {
                        'lat': float(ep_r['Latitude']),
                        'lon': float(ep_r['Longitude']),
                        'tilt': mech + dig
                    }
                    if 'gNodeB ID' in ep_r:
                        pci_gnodeb_map[p] = str(ep_r['gNodeB ID']).strip()
                except: continue

        # --- EXTRACT FEATURES (Max per sample) ---
        
        # C1 Tilt & C2 Distance
        max_tilt = 0
        max_dist = 0
        for _, u_r in df_up.iterrows():
            s_pci = u_r.get('5G KPI PCell RF Serving PCI')
            if pd.notna(s_pci) and s_pci in pci_ep_map:
                ep = pci_ep_map[s_pci]
                max_tilt = max(max_tilt, ep['tilt'])
                if pd.notna(u_r.get('Latitude')):
                    d = haversine(u_r['Latitude'], u_r['Longitude'], ep['lat'], ep['lon'])
                    max_dist = max(max_dist, d)
        
        # C3 Neighbor
        # (Actually C3 is default, so we don't strictly need to calc it if it's the loser)
        
        # C4 Interference
        non_col_strong = False
        n1_pci_c = 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI'
        n1_rsrp_c = 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]'
        for _, u_r in df_up.iterrows():
             n_rsrp = pd.to_numeric(u_r.get(n1_rsrp_c), errors='coerce')
             s_pci = u_r.get('5G KPI PCell RF Serving PCI')
             n_pci = u_r.get(n1_pci_c)
             
             if pd.notna(n_rsrp) and n_rsrp > -105:
                 if pd.notna(s_pci) and pd.notna(n_pci):
                     try:
                         s_id = str(pci_gnodeb_map.get(int(s_pci)))
                         n_id = str(pci_gnodeb_map.get(int(n_pci)))
                         if s_id and n_id and s_id != n_id and s_id!='None' and n_id!='None':
                             non_col_strong = True
                     except: pass

        # C5 Handover
        ho_count = df_up['5G KPI PCell RF Serving PCI'].nunique()

        # C6 Collision
        collision = False
        for _, u_r in df_up.iterrows():
             s_pci = u_r.get('5G KPI PCell RF Serving PCI')
             n_pci = u_r.get(n1_pci_c)
             try:
                 if int(s_pci) % 30 == int(n_pci) % 30:
                     collision = True
             except: pass
             
        # C7 Speed
        max_speed = df_up['GPS Speed (km/h)'].max()
        if pd.isna(max_speed): max_speed = 0
        
        # C8 RBs
        mean_rbs = df_up['5G KPI PCell Layer1 DL RB Num (Including 0)'].mean()
        if pd.isna(mean_rbs): mean_rbs = 200

        # --- APPLY PRIORITY LOGIC (The Decision Tree) ---
        
        # Tier 1: Absolutes
        if max_speed > 40:
            pred = 'C7'
        elif max_dist > 1.0:
            pred = 'C2'
        elif mean_rbs < 160:
            pred = 'C8'
        elif ho_count > 2:
            pred = 'C5' # Placement of C5? Usually high priority if frequent HOs
            # Let's check where C5 fits. Users usually hate HOs. 
            # In absence of data, placing it high is safer.
            
        # Tier 2: Severe Interference
        elif non_col_strong:
            pred = 'C4'
            
        # Tier 3: Optimization
        elif max_tilt > 40: # C1
            pred = 'C1'
        elif collision: # C6
            pred = 'C6' # C6 is rare, usually if C4 is not triggered
            
        # Tier 4: Catch-all
        else:
            pred = 'C3'
            
        predictions.append({'ID': row_id, 'Root_Cause': pred})

        # Save
    out_df = pd.DataFrame(predictions)
    out_df.to_csv(output_file, index=False)
    print(f"Saved {len(out_df)} predictions to {output_file}")

def extract_features(df_up, df_ep):
    # --- PREPROCESSING ---
    cols_to_numeric = [
        '5G KPI PCell RF Serving SS-RSRP [dBm]',
        '5G KPI PCell Layer1 DL RB Num (Including 0)',
        '5G KPI PCell RF Serving PCI',
        'GPS Speed (km/h)',
        'Latitude', 'Longitude'
    ]
    for c in cols_to_numeric:
        if c in df_up.columns:
            df_up[c] = pd.to_numeric(df_up[c], errors='coerce')

    # Build Maps
    pci_ep_map = {}
    pci_gnodeb_map = {}
    if not df_ep.empty:
        for _, ep_r in df_ep.iterrows():
            try:
                p = int(ep_r['PCI'])
                mech = float(ep_r['Mechanical Downtilt']) if 'Mechanical Downtilt' in ep_r else 0
                dig = float(ep_r['Digital Tilt']) if 'Digital Tilt' in ep_r else 0
                if dig == 255: dig = 6.0
                
                pci_ep_map[p] = {
                    'lat': float(ep_r['Latitude']),
                    'lon': float(ep_r['Longitude']),
                    'tilt': mech + dig
                }
                if 'gNodeB ID' in ep_r:
                    pci_gnodeb_map[p] = str(ep_r['gNodeB ID']).strip()
            except: continue

    # --- EXTRACT FEATURES ---
    max_tilt = 0
    max_dist = 0
    for _, u_r in df_up.iterrows():
        s_pci = u_r.get('5G KPI PCell RF Serving PCI')
        if pd.notna(s_pci) and s_pci in pci_ep_map:
            ep = pci_ep_map[s_pci]
            max_tilt = max(max_tilt, ep['tilt'])
            if pd.notna(u_r.get('Latitude')):
                d = haversine(u_r['Latitude'], u_r['Longitude'], ep['lat'], ep['lon'])
                max_dist = max(max_dist, d)
    
    non_col_strong = False
    n1_pci_c = 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI'
    n1_rsrp_c = 'Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]'
    for _, u_r in df_up.iterrows():
            n_rsrp = pd.to_numeric(u_r.get(n1_rsrp_c), errors='coerce')
            s_pci = u_r.get('5G KPI PCell RF Serving PCI')
            n_pci = u_r.get(n1_pci_c)
            
            if pd.notna(n_rsrp) and n_rsrp > -105:
                if pd.notna(s_pci) and pd.notna(n_pci):
                    try:
                        s_id = str(pci_gnodeb_map.get(int(s_pci)))
                        n_id = str(pci_gnodeb_map.get(int(n_pci)))
                        if s_id and n_id and s_id != n_id and s_id!='None' and n_id!='None':
                            non_col_strong = True
                    except: pass

    ho_count = df_up['5G KPI PCell RF Serving PCI'].nunique()

    collision = False
    for _, u_r in df_up.iterrows():
            s_pci = u_r.get('5G KPI PCell RF Serving PCI')
            n_pci = u_r.get(n1_pci_c)
            try:
                if int(s_pci) % 30 == int(n_pci) % 30:
                    collision = True
            except: pass
            
    max_speed = df_up['GPS Speed (km/h)'].max()
    if pd.isna(max_speed): max_speed = 0
    
    mean_rbs = df_up['5G KPI PCell Layer1 DL RB Num (Including 0)'].mean()
    if pd.isna(mean_rbs): mean_rbs = 200
    
    # Switches & PingPong
    pcis = df_up['5G KPI PCell RF Serving PCI'].tolist()
    num_switches = 0
    ping_pong = False
    history = []
    if len(pcis) > 0:
        last_pci = pcis[0]
        history = [last_pci]
        for p in pcis[1:]:
            if p != last_pci:
                num_switches += 1
                history.append(p)
                last_pci = p
    
    if len(history) >= 3:
        if history[-1] == history[-3]: 
             ping_pong = True
             
    # Neighbor Better
    neighbor_better = False
    for _, u_r in df_up.iterrows():
         s_rsrp = pd.to_numeric(u_r.get('5G KPI PCell RF Serving SS-RSRP [dBm]'), errors='coerce')
         n_rsrp = pd.to_numeric(u_r.get('Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]'), errors='coerce')
         if pd.notna(s_rsrp) and pd.notna(n_rsrp):
             if n_rsrp > s_rsrp + 3:
                 neighbor_better = True
    
    return {
        'max_tilt': max_tilt,
        'max_dist': max_dist,
        'non_col_strong': non_col_strong,
        'ho_count': ho_count,
        'collision': collision,
        'max_speed': max_speed,
        'mean_rbs': mean_rbs,
        'num_switches': num_switches,
        'ping_pong': ping_pong,
        'neighbor_better': neighbor_better
    }

def classify_sample(features):
    # --- PREDICT ---
    # Adjusted Thresholds & Re-ordered Priority
    # Tier 1: Absolute Physical Constraints
    if features['max_speed'] > 40: return 'C7'
    elif features['max_dist'] > 1.0: return 'C2'
    
    # Tier 2: Specific Root Causes (Interference, Tilt, Collision)
    # These must be checked BEFORE generic symptoms like RBs/HO
    elif features['non_col_strong']: return 'C4'
    elif features['max_tilt'] > 40: return 'C1'
    elif features['collision']: return 'C6'
    
    # Tier 3: Optimization Symptoms (Broad Rules)
    # Now safe to use broader thresholds because we ruled out C4/C1 above
    elif features['mean_rbs'] < 170: return 'C8' # Adjusted from 160
    elif features['ho_count'] >= 2: return 'C5' # Adjusted from > 2
    
    # Tier 4: Catch-all
    else: return 'C3'

def validate_on_train():
    from seperate_values import get_all_data_iterator
    
    print("Validating on train.csv...")
    correct = 0
    total = 0
    by_class_correct = {}
    by_class_total = {}
    
    for index, raw_row, df_up, df_ep in get_all_data_iterator():
        # Get Truth
        truth_match = re.search(r'(C[1-8])', str(raw_row['answer']))
        if not truth_match: continue
        true_label = truth_match.group(1)
        
        # Extract & Predict
        feats = extract_features(df_up, df_ep)
        pred = classify_sample(feats)
        
        # --- CHECK ---
        total += 1
        by_class_total[true_label] = by_class_total.get(true_label, 0) + 1
        
        if pred == true_label:
            correct += 1
            by_class_correct[true_label] = by_class_correct.get(true_label, 0) + 1
            
    print(f"\nOverall Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    print("\nAccuracy by Class:")
    metrics_data = []
    classes = sorted(by_class_total.keys())
    for c in classes:
        c_corr = by_class_correct.get(c, 0)
        c_tot = by_class_total.get(c, 0)
        acc = (c_corr/c_tot)*100 if c_tot > 0 else 0
        print(f"{c}: {c_corr}/{c_tot} ({acc:.2f}%)")
        metrics_data.append({'Class': c, 'Accuracy': acc, 'Count': c_tot})
        
    return metrics_data

if __name__ == "__main__":
    # If run directly to validate
    print("--- Train Validation ---")
    validate_on_train()
    print("\n--- Generating Test Predictions ---")
    run_classification()
