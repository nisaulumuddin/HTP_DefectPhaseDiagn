import os
import pathlib

import ovito.io
import pandas as pd
from pyroxene.core.workflow.datasource import PresetMaterialsProjectFileSource
from pyroxene.structure.pure import W, Mg, b_B, graph_C, Cu, dia_C, bla_P, a_S, Mn, a_Ga, a_As, In, b_Sn, Cl, a_La, \
    a_Sm, Pa, a_U, a_Np, a_Pu
from ruamel.yaml import YAML

yaml = YAML(typ='safe')

def collect_structure(listdir):
    # listdir = os.listdir(root_folder)
    structure_data = []
    for folder in pathlib.Path(listdir):
        data_path = folder / "bi_minimised.data.yaml"
        structure_file = folder/ "bi_minimised.data"
        
        ovito_file = ovito.io.import_file(structure_file)
        data_ovito = ovito_file.compute()
        
        data = read_yaml(data_path)

        result = {
            "folder": folder,
            "E_cell": data["pe"],
            "atoms": data["atoms"],
            "area_A2": data["lx"] * data["ly"]
        }

        type_property = data_ovito.particles['Particle Type']
        id_to_name = { t.id: t.name for t in type_property.types}
        unique_type, count_of_that_type = np.unique(data_ovito.particles["Particle Type"].array, return_counts=True)
        type_count = { id_to_name[type_id]: int(count) for type_id, count in zip(unique_type, count_of_that_type)}
        
        for type in type_count:
            result[f"{type}_atoms"] = int(type_count[type])  # Using name as key
        
        structure_data.append(result)
    df_structure = pd.DataFrame(data=structure_data)
    return df_structure, structure_data

def get_ef(path: pathlib.Path, pure_df: pd.DataFrame, elem_list: list[str]):
    with open(path, 'r') as file:
        data = yaml.load(file)

        f = ovito.io.import_file(path.with_suffix(""))
        a7b6_types = (f.compute().particles_.particle_types_.array - 1)

        from collections import Counter

        def count_unique_values(array):
            v = dict(Counter(array))
            return {int(dk): dv for dk, dv in v.items()}

        unique_counts = count_unique_values(a7b6_types)
        fe_dict = {}
        for k, v in unique_counts.items():
            element_to_query = elem_list[k]
            fe_dict[k] = float(pure_df.where(pure_df["e1"] == element_to_query).dropna()["Ef"].values[0])
        ref_en = 0
        for k, v in unique_counts.items():
            ref_en += fe_dict[k] * v
        a7b6_fe = data[0]["pe"] - ref_en
        return a7b6_fe / f.compute().particles_.count


def read_yaml(path):
    with open(path, mode="rb") as f:
        y = yaml.load(f)
    return y[0]


# Get all .out files from subdirectory.
def get_out_files(directory):
    out_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.out'):
                out_files.append(os.path.join(root, file))
    return out_files


# Parse .out files and extract data.
def parse_out_file(file_path):
    yaml = YAML(typ='safe')
    with open(file_path, 'r') as file:
        data = yaml.load(file)
    return data


# Populate DataFrame with the values.
def create_dataframe(out_files):
    records = []
    for file in out_files:
        data = parse_out_file(file)
        for entry in data:
            e1 = entry.get('e1')
            eatoms = entry.get('eatoms')
            atoms = entry.get('atoms')
            cella = entry.get('cella')
            cellb = entry.get('cellb')
            cellc = entry.get('cellc')
            cellalpha = entry.get('cellalpha')
            cellbeta = entry.get('cellbeta')
            cellgamma = entry.get('cellgamma')
            Ef = eatoms / atoms if atoms else None
            records.append({
                'e1': e1.strip(),
                'eatoms': eatoms,
                'atoms': atoms,
                'cella': cella,
                'cellb': cellb,
                'cellc': cellc,
                'cellalpha': cellalpha,
                'cellbeta': cellbeta,
                'cellgamma': cellgamma,
                'Ef': Ef
            })
    return pd.DataFrame(records)


def stp_structure(element: str) -> PresetMaterialsProjectFileSource:
    table = {
        "H": None,
        "He": None,
        "Li": W,
        "Be": Mg,
        "B": b_B,
        "C": graph_C,
        "N": None,
        "O": None,
        "F": None,
        "Ne": None,
        "Na": W,
        "Mg": Mg,
        "Al": Cu,
        "Si": dia_C,
        "P": bla_P,
        "S": a_S,
        "Cl": None,
        "Ar": None,
        "K": W,
        "Ca": Cu,
        "Sc": Mg,
        "Ti": Mg,
        "V": W,
        "Cr": W,
        "Mn": Mn,
        "Fe": W,
        "Co": Mg,
        "Ni": Cu,
        "Cu": Cu,
        "Zn": Mg,
        "Ga": a_Ga,
        "Ge": dia_C,
        "As": a_As,
        "Se": None,  # Gamma selenium
        "Br": None,
        "Kr": None,
        "Rb": W,
        "Sr": Cu,
        "Y": Mg,
        "Zr": Mg,
        "Nb": W,
        "Mo": W,
        "Tc": Mg,
        "Ru": Mg,
        "Rh": Cu,
        "Pd": Cu,
        "Ag": Cu,
        "Cd": Mg,
        "In": In,
        "Sn": b_Sn,
        "Sb": a_As,
        "Te": None,  # Gamma selenium
        "I": Cl,
        "Xe": None,
        "Cs": W,
        "Ba": W,
        "Lu": Mg,
        "Hf": Mg,
        "Ta": W,
        "W": W,
        "Re": Mg,
        "Os": Mg,
        "Ir": Cu,
        "Pt": Cu,
        "Au": Cu,
        "Hg": None,
        "Tl": Mg,
        "Pb": Cu,
        "Bi": a_As,
        "Po": None,  # Alpha polonium
        "At": None,
        "Rn": None,
        "Fr": None,
        "Ra": W,
        "Lr": None,
        "Rf": None,
        "Db": None,
        "Sg": None,
        "Bh": None,
        "Hs": None,
        "Mt": None,
        "Ds": None,
        "Rg": None,
        "Cn": None,
        "Nh": None,
        "Fl": None,
        "Mc": None,
        "Lv": None,
        "Ts": None,
        "Og": None,
        "La": a_La,
        "Ce": a_La,
        "Pr": a_La,
        "Nd": a_La,
        "Pm": a_La,
        "Sm": a_Sm,
        "Eu": W,
        "Gd": Mg,
        "Tb": Mg,
        "Dy": Mg,
        "Ho": Mg,
        "Er": Mg,
        "Tm": Mg,
        "Yb": Cu,
        "Ac": Cu,
        "Th": Cu,
        "Pa": Pa,
        "U": a_U,
        "Np": a_Np,
        "Pu": a_Pu,
        "Am": a_La,
        "Cm": a_La,
        "Bk": a_La,
        "Cf": a_La,
        "Es": Cu,
        "Fm": None,
        "Md": None,
        "No": None,
    }
    return table[element]


def get_pure_mu(pure_df, type_):
    return float(pure_df.where(pure_df["e1"] == type_).dropna()["Ef"].values[0])
