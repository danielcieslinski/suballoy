from urllib.request import Request, urlopen
import json
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm 
from pandas import DataFrame, read_csv, read_pickle

# Get api key
with open('.secrets/mat_proj_api_key.txt', 'r') as f:
    API_KEY = f.read().strip()

# Utils
# --------

def split_list_into_chunks(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]

# Download data from Materials Project
# --------

def get_all_materials_ids():
    req = Request('https://www.materialsproject.org/rest/v1/materials/mids', headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    data = json.loads(webpage.decode())
    mat_ids = data['response']
    return mat_ids

def fetch_materials(materials_ids, props=['material_id', 'structure', 'formation_energy_per_atom', 'elasticity.K_VRH', 'elasticity.G_VRH']):
    # https://docs.materialsproject.org/downloading-data/api-endpoint-documentation/summary
    # (!) https://hackingmaterials.lbl.gov/matminer/example_bulkmod.html
    chunks = list(split_list_into_chunks(materials_ids, 1000))
    data = []

    with MPRester(API_KEY) as client:
        for chunk in tqdm(chunks):
            criteria={'material_id': {'$in':chunk}}
            results = client.query(criteria, properties=props)
            for mat in results:
                data.append(mat)
                
    return data

def load_df(df_path='res/mp_df.pickle'):
    return read_pickle(df_path)

if __name__ == "__main__":
    mat_ids = get_all_materials_ids()
    data = fetch_materials(mat_ids)
    df = DataFrame(data)
    df.to_pickle('res/mp_df.pickle')