from utils import read_data
from mat_proj import load_df

defected_data = read_data('res/kvrh_defect_3.pickle')
baseline_data, _ = read_data('res/kvrh_parity.pickle')

df = load_df()
kvrh_df = df[~df['elasticity.K_VRH'].isnull()]

def biggest_outlier(defected_data, baseline_data):
    candidate_index, candidate_val_split = 0, 0
    for i, D in enumerate(zip(defected_data, baseline_data)):
        a,b = D
        x = abs(a-b)
        if x > candidate_val_split:
            candidate_val_split = x
            candidate_index = i
    return candidate_index, candidate_val_split

def find_outliers(df, defected_data, baseline_data):
    out = {}
    for k in defected_data:
        idx, val = biggest_outlier(defected_data[k], baseline_data)
        out[k] = (baseline_data[idx], defected_data[k][idx], df.iloc[idx]['structure'])
    return out

find_outliers(kvrh_df, defected_data, baseline_data)
