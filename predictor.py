from megnet.utils.models import load_model, AVAILABLE_MODELS
from pymatgen.core import Structure, Lattice
from pymatgen.ext.matproj import MPRester

class Predictor:
    def __init__(self):
        self.log_kvrh_model = load_model("logK_MP_2019")
        self.log_gvrh_model = load_model("logG_MP_2019")
        self.eform_model = load_model("Eform_MP_2019")

    def predict_kvrh(self, structure):
        return 10 ** self.log_kvrh_model.predict_structure(structure).ravel()[0]
    
    def predict_gvrh(self, structure):
        return 10 ** self.log_gvrh_model.predict_structure(structure).ravel()[0]
    
    def predict_eform(self, structure):
        return self.eform_model.predict_structure(structure).ravel()[0]
    
    def predict_all(self, structure):
        return self.predict_kvrh(structure), self.predict_gvrh(structure), self.predict_eform(structure)
    
    def available_models(self):
        return AVAILABLE_MODELS
    

def test(predictor):
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.2), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    # print(predictor.available_models())
    print(predictor.predict_kvrh(s))
    print(predictor.predict_gvrh(s))
    print(predictor.predict_eform(s))

if __name__ == "__main__":
    predictor = Predictor()
    test(predictor)