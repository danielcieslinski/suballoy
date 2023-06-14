from predictor import Predictor
from pymatgen.core import Structure, Lattice
import os 

def print_res(data):
    print('K_VRH', data[0], '(GPa)')
    print('G_VRH', data[1], '(GPa)')
    print('E_form', data[2], '(eV/atom)')
    
def pred_all(predictor):
    for ffile in os.listdir('validation_samples'):
        if ffile.endswith('.cif'):
            print('File:', ffile)
            print_res(predictor.predict_all(Structure.from_file('validation_samples/' + ffile)))
            print()

predictor = Predictor()
pred_all(predictor)
