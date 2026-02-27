import pandas as pd
import io
import re

def parse_network_data(raw_text):
    """
    Parse network data from raw text string.
    Supports Type A (Phase 1/Train/Phase 2 EP-first) and Type B (Phase 2 Markdown) formats.
    Returns standardized (df_up, df_ep).
    """
    
    # Define generic lookahead pattern for any next section start
    # Matches "\n\n" followed by any known header start
    next_section_lookahead_a = r"(?=\n\n(?:User plane|Engeneering|Signaling|Given:)|$)" 
    # Type B lookahead: Matches "**" marking new section or End
    next_section_lookahead_b = r"(?=\n\n\*\*|\n\*\*|Given:|$)"

    # --- 1. Try Type A (Original Format) ---
    # Note: "Engeneering" typo is present in data
    user_plane_pattern_a = r"User plane drive test data as follows[：:]\s*(Timestamp\|.*?)" + next_section_lookahead_a
    eng_params_pattern_a = r"Engeneering parameters data as follows[：:]\s*(gNodeB ID\|.*?)" + next_section_lookahead_a
    
    user_plane_match_a = re.search(user_plane_pattern_a, raw_text, re.DOTALL)
    eng_params_match_a = re.search(eng_params_pattern_a, raw_text, re.DOTALL)
    
    if user_plane_match_a:
        try:
            data_str = user_plane_match_a.group(1).strip()
            # Handle case where regex might claim too much if lookahead strictness varies
            # But usually | separated data is robust
            df_up = pd.read_csv(io.StringIO(data_str), sep='|')
            
            df_ep = pd.DataFrame()
            if eng_params_match_a:
                data_str = eng_params_match_a.group(1).strip()
                df_ep = pd.read_csv(io.StringIO(data_str), sep='|')
            
            return df_up, df_ep
        except Exception:
            # If parsing fails, fall through to try Type B logic (unlikely but safe)
            pass

    # --- 2. Try Type B (Markdown Table / Mixed Format) ---
    # Header: | Time | UE | ...
    user_plane_pattern_b = r"\*\*Drive Test Data\*\*\s*(\|.*?)" + next_section_lookahead_b
    eng_params_pattern_b = r"\*\*Parameter Data\*\*\s*(\|.*?)" + next_section_lookahead_b
    
    user_plane_match_b = re.search(user_plane_pattern_b, raw_text, re.DOTALL)
    
    if user_plane_match_b:
        data_str = user_plane_match_b.group(1).strip()
        
        # Remove markdown alignment row (e.g. |:---:|---:|)
        lines = data_str.split('\n')
        # Filter out separator lines (contain only |-:)
        lines = [line for line in lines if not re.match(r'^\s*\|?[:\-\|]+\s*$', line)]
        
        try:
            cleaned_data = '\n'.join(lines)
            df_up = pd.read_csv(io.StringIO(cleaned_data), sep=r'\s*\|\s*', engine='python')
            df_up = df_up.loc[:, ~df_up.columns.str.contains('^Unnamed')]
            
            # Standardize Columns
            column_map_up = {
                'Time': 'Timestamp',
                'Serving PCI': '5G KPI PCell RF Serving PCI',
                'Serving RSRP(dBm)': '5G KPI PCell RF Serving SS-RSRP [dBm]',
                'Serving SINR(dB)': '5G KPI PCell RF Serving SS-SINR [dB]',
                'Throughput(Mbps)': '5G KPI PCell Layer2 MAC DL Throughput [Mbps]',
                'RB/slot': '5G KPI PCell Layer1 DL RB Num (Including 0)',
            }
            # Neighbor mapping
            for i in range(1, 6):
                column_map_up[f'Neighbor {i} PCI'] = f'Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI'
                column_map_up[f'Neighbor {i} RSRP(dBm)'] = f'Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]'
                
            df_up = df_up.rename(columns=column_map_up)
            
            # Defaults
            if 'GPS Speed (km/h)' not in df_up.columns:
                df_up['GPS Speed (km/h)'] = 0
            
            # EP Type B
            df_ep = pd.DataFrame()
            eng_params_match_b = re.search(eng_params_pattern_b, raw_text, re.DOTALL)
            if eng_params_match_b:
                data_str = eng_params_match_b.group(1).strip()
                lines = data_str.split('\n')
                lines = [line for line in lines if not re.match(r'^\s*\|?[:\-\|]+\s*$', line)]
                cleaned_data = '\n'.join(lines)
                df_ep = pd.read_csv(io.StringIO(cleaned_data), sep=r'\s*\|\s*', engine='python')
                df_ep = df_ep.loc[:, ~df_ep.columns.str.contains('^Unnamed')]
                
                column_map_ep = {
                    'Mech Tilt(deg)': 'Mechanical Downtilt',
                    'Elec Tilt(deg)': 'Electrical Downtilt',
                    'Azimuth(deg)': 'Azimuth'
                }
                df_ep = df_ep.rename(columns=column_map_ep)
                
            return df_up, df_ep
            
        except Exception as e:
            print(f"Warning: Failed to parse Type B table: {e}")
            return pd.DataFrame(), pd.DataFrame()

    return pd.DataFrame(), pd.DataFrame()


def load_data():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'train.csv')
    df_train = pd.read_csv(csv_path)
    sample_text = df_train['question'].iloc[0]
    user_plane, eng_params = parse_network_data(sample_text)
    return user_plane, eng_params

def get_all_data_iterator():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'train.csv')
    df_train = pd.read_csv(csv_path)
    
    for index, raw_row in df_train.iterrows():
        sample_text = raw_row['question']
        try:
            user_plane, eng_params = parse_network_data(sample_text)
            yield index, raw_row, user_plane, eng_params
        except Exception:
            continue

if __name__ == "__main__":
    pass