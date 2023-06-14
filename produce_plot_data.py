from predictor import Predictor
from mat_proj import load_df
from pymatgen.core import Structure, Lattice
from mendeleev import element
from utils import write_data

def make_elements():
    elements = [element(i).symbol for i in range(1,93)]
    return list(map(str, elements))

# Parity data
def produce_kvrh_parity_data(df, predictor):
    predicted, actual = [], []
    kvrh_df = df[~df['elasticity.K_VRH'].isnull()]

    for i in range(len(kvrh_df)):
        row = kvrh_df.iloc[i]
        structure = row['structure']
        kvrh = row['elasticity.K_VRH']
        print(f'{i}/{len(kvrh_df)}')
        try:
            predicted_kvrh = predictor.predict_kvrh(structure)
            predicted.append(predicted_kvrh)
            actual.append(kvrh)
        except: pass

    write_data('res/kvrh_parity.pickle', (predicted, actual))
    
def produce_gvrh_parity_data(df, predictor):
    predicted, actual = [], []
    gvrh_df = df[~df['elasticity.G_VRH'].isnull()]

    for i in range(len(gvrh_df)):
        row = gvrh_df.iloc[i]
        structure = row['structure']
        gvrh = row['elasticity.G_VRH']
        print(f'{i}/{len(gvrh_df)}')
        try:
            predicted_gvrh = predictor.predict_gvrh(structure)
            predicted.append(predicted_gvrh)
            actual.append(gvrh)
        except: pass

    write_data('res/gvrh_parity.pickle', (predicted, actual))

def produce_eform_parity_data(df, predictor):
    predicted, actual = [], []
    eform_df = df[~df['formation_energy_per_atom'].isnull()]

    for i in range(len(eform_df)):
        row = eform_df.iloc[i]
        structure = row['structure']
        eform = row['formation_energy_per_atom']
        print(f'{i}/{len(eform_df)}')
        try:
            predicted_eform = predictor.predict_eform(structure)
            predicted.append(predicted_eform)
            actual.append(eform)
        except: pass

    write_data('res/eform_parity.pickle', (predicted, actual))


### Defects
def produce_kvrh_defect_data(df, predictor, elems):
    res = {elem:[] for elem in elems}
    kvrh_df = df[~df['elasticity.K_VRH'].isnull()]
    failed = 0

    for i in range(len(kvrh_df)):
        row = kvrh_df.iloc[i]
        structure = row['structure'] * (3,3,3)
        for elem in elems:
            structure[0] = elem
            print(f'{elem}:{i}/{len(kvrh_df)}')
            try:
                predicted_kvrh = predictor.predict_kvrh(structure)
                res[elem].append(predicted_kvrh)
            except: failed+=1

    write_data('res/kvrh_defect_3.pickle', res)
    print(f'Failed: {failed}')

def produce_gvrh_defect_data(df, predictor, elems):
    res = {elem:[] for elem in elems}
    gvrh_df = df[~df['elasticity.G_VRH'].isnull()]

    for i in range(len(gvrh_df)):
        row = gvrh_df.iloc[i]
        structure = row['structure'] * (3,3,3)
        for elem in elems:
            structure[0] = elem
            print(f'{elem}:{i}/{len(gvrh_df)}')
            try:
                predicted_gvrh = predictor.predict_gvrh(structure)
                res[elem].append(predicted_gvrh)
            except: pass

    write_data('res/gvrh_defect_3.pickle', res)

def produce_eform_defect_data(df, predictor, elems):
    res = {elem:[] for elem in elems}
    eform_df = df[~df['formation_energy_per_atom'].isnull()]

    #for i in range(len(eform_df)):
    for i in range(20_000):
        row = eform_df.iloc[i]
        structure = row['structure'] * (3,3,3)
        for elem in elems:
            structure[0] = elem
            print(f'{elem}:{i}/{len(eform_df)}')
            try:
                predicted_eform = predictor.predict_eform(structure)
                res[elem].append(predicted_eform)
            except: pass

    write_data('res/eform_defect_3_20k.pickle', res)

# ptable defects
def make_defect_for_all_elements(predictor, base_structure, elements):
    res = {'k_vrh': [], 'g_vrh': [], 'eform': [], 'elem': []}
    # elements = make_elements()
    for elem in elements:
        structure = base_structure.copy()
        structure[0] = elem
        try:
            res['k_vrh'].append(predictor.predict_kvrh(structure))
            res['g_vrh'].append(predictor.predict_gvrh(structure))
            res['eform'].append(predictor.predict_eform(structure))
            res['elem'].append(elem)
        except: print(f'Failed for {elem} and structure {structure}')
    return res

def produce_ptable_defect_data(predictor):
    data = {}
    elements = make_elements()
    # ext_tups = [(2,2,2), (4,4,4), (8,8,8)]
    ext_tups = [(3,3,3)]
    ni_fcc = Structure.from_spacegroup('Fm-3m', Lattice.cubic(3.51), ['Ni'], [[0]*3]) 
    al_fcc = Structure.from_spacegroup('Fm-3m', Lattice.cubic(4.04), ['Al'], [[0]*3]) 
    au_fcc = Structure.from_spacegroup('Fm-3m', Lattice.cubic(4.17), ['Au'], [[0]*3]) 
    mo_bcc = Structure.from_spacegroup('Im-3m', Lattice.cubic(3.17), ['Mo'], [[0]*3]) 

    for ext in ext_tups:
        data[ext[0]] = {}

        data[ext[0]]['ni_fcc'] = make_defect_for_all_elements(predictor, ni_fcc * ext, elements)
        data[ext[0]]['al_fcc'] = make_defect_for_all_elements(predictor, al_fcc * ext, elements)
        data[ext[0]]['au_fcc'] = make_defect_for_all_elements(predictor, au_fcc * ext, elements)
        data[ext[0]]['mo_bcc'] = make_defect_for_all_elements(predictor, mo_bcc * ext, elements)

    write_data('res/ptable_defect_3.pickle', data)


# delta ptable plots
def sub_value_in_nodefected(predictor, defected_data, pure_structure):
    # For each property and defected structure substitute this property of pure structure
    pure_kvrh = predictor.predict_kvrh(pure_structure)
    pure_gvrh = predictor.predict_gvrh(pure_structure)
    pure_eform = predictor.predict_eform(pure_structure)
    defected_data['k_vrh'] = list(map(lambda x: x - pure_kvrh, defected_data['k_vrh']))
    defected_data['g_vrh'] = list(map(lambda x: x - pure_gvrh, defected_data['g_vrh']))
    defected_data['eform'] = list(map(lambda x: x - pure_eform, defected_data['eform']))

    return defected_data
    

def produce_delta_ptable_data(predictor):
    data = {}
    elements = make_elements()
    ext_tups = [(2,2,2), (3,3,3), (8,8,8)]
    ni_fcc = Structure.from_spacegroup('Fm-3m', Lattice.cubic(3.51), ['Ni'], [[0]*3]) 
    al_fcc = Structure.from_spacegroup('Fm-3m', Lattice.cubic(4.04), ['Al'], [[0]*3]) 
    au_fcc = Structure.from_spacegroup('Fm-3m', Lattice.cubic(4.17), ['Au'], [[0]*3]) 
    mo_bcc = Structure.from_spacegroup('Im-3m', Lattice.cubic(3.17), ['Mo'], [[0]*3]) 

    for ext in ext_tups:
        data[ext[0]] = {}

        data[ext[0]]['ni_fcc'] = sub_value_in_nodefected(predictor, make_defect_for_all_elements(predictor, ni_fcc * ext, elements), ni_fcc * ext)
        data[ext[0]]['al_fcc'] = sub_value_in_nodefected(predictor, make_defect_for_all_elements(predictor, al_fcc * ext, elements), al_fcc * ext)
        data[ext[0]]['au_fcc'] = sub_value_in_nodefected(predictor, make_defect_for_all_elements(predictor, au_fcc * ext, elements), au_fcc * ext)
        data[ext[0]]['mo_bcc'] = sub_value_in_nodefected(predictor, make_defect_for_all_elements(predictor, mo_bcc * ext, elements), mo_bcc * ext)

    write_data('res/ptable_delta_defects.pickle', data)

# saturation plot
def produce_saturation_plot_data(predictor):
    full_dict = {}
    elements = ['H', 'Mn', 'Rb']
    
    mo_bcc = Structure.from_spacegroup('Im-3m', Lattice.cubic(3.17), ['Mo'], [[0]*3]) 
    tups = [(k,k,k) for k in range(2,  9)]

    for prop, func in [('k_vrh', predictor.predict_kvrh), ('g_vrh', predictor.predict_gvrh), ('eform', predictor.predict_eform)]:
        data = {}
        for elem in elements:
            data[elem] = []
            for tup in tups:
                struct = mo_bcc.copy() * tup
                struct[0] = elem
                res = func(struct)
                data[elem].append([struct.num_sites, res])
        full_dict[prop] = data
    
    write_data('res/saturation.pickle', full_dict)


if __name__ == "__main__":
    df = load_df()
    predictor = Predictor()
    produce_kvrh_parity_data(df, predictor)
    produce_gvrh_parity_data(df, predictor)
    produce_eform_parity_data(df, predictor)
    produce_kvrh_defect_data(df, predictor, ['H', 'Mn', 'Rb'])
    produce_gvrh_defect_data(df, predictor, ['H', 'Mn', 'Rb'])
    produce_eform_defect_data(df, predictor, ['H', 'Mn', 'Rb'])
    produce_ptable_defect_data(predictor)
    produce_delta_ptable_data(predictor)
    produce_saturation_plot_data(predictor)