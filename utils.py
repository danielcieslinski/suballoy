import pickle

# Utils
def read_data(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def write_data(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def save_cif(struct, path):
    CifWriter(struct, symprec=1e-6).write_file(path)