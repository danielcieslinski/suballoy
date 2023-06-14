import numpy as np
from utils import read_data

def safe_log10(a, b):
    a, b = np.log10(a), np.log10(b)
    f = lambda x: np.isnan(x) or np.isinf(x)
    fil = filter(lambda x: not(f(x[0]) or f(x[1])), zip(a, b))
    a, b = list(zip(*fil))
    return a, b

def mae(x, y):
    return np.mean(np.abs(np.asarray(x)-np.asarray(y)))

pred_kvrh, actual_kvrh = read_data('res/kvrh_parity.pickle')
print('KVRH [mae] (GPa):', mae(pred_kvrh, actual_kvrh))
print('KVRH [mae] log10(GPa):', mae(*safe_log10(pred_kvrh, actual_kvrh)))
print(f'Number of records: {len(actual_kvrh)} ')

pred_gvrh, actual_gvrh = read_data('res/gvrh_parity.pickle')
print('GVRH [mae] (GPa):', mae(pred_gvrh, actual_gvrh))
print('GVRH [mae] log10(GPa):', mae(*safe_log10(pred_gvrh, actual_gvrh)))
print(f'Number of records: {len(actual_gvrh)} ')

pred_eform, actual_eform = read_data('res/eform_parity.pickle')
print('Eform [mae] (eV/atom):', mae(pred_eform, actual_eform))
print(f'Number of records: {len(actual_eform)} ')

