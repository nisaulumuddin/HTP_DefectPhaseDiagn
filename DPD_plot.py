import matplotlib.pyplot as plt
import numpy as np
import os
from common import get_ef, get_out_files, create_dataframe, read_yaml, get_pure_mu 
from ruamel.yaml import YAML
import ovito.io
from collections import defaultdict
import pandas as pd
from itertools import combinations
import pathlib

# Defect Phase Diagram (DPD) 

# Prepared for high-throughput calculation of mu-phase defects formation energies in A6B6 and A7B6. 
# The A6B7 and A7B6 mu-phase as treated as 2 separate stable phases. 

# The upper and lower limits of chemical potentials mu_A and mu_B where the mu-phase is stable are defined as follows
# DPD of A6B7:
# poor A limit: C14 Laves phase  + A6B7
# rich A limit: A6B7 + A7B6

# DPD of A7B6:
# poor A limit: A6B7 + A7B6
# rich A limit: A7B6 + A(BCC)


def collect_structure(listdir,pristine_data = None, pristine_keyword = None  ): 
    # Objective: Collect structure data from .yaml file
    
    structure_data = []
    for folder in listdir:
        print("collecting data from " + str(folder))
        data_path = pathlib.Path(folder) / "bi_minimised.data.yaml"
        structure_file =pathlib.Path(folder) / "bi_minimised.data"
        
        ovito_file = ovito.io.import_file(structure_file)
        data_ovito = ovito_file.compute()
        
        data = read_yaml(data_path)
        basename =  os.path.basename(folder)
        result = {
            "folder": basename,
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
            if pristine_data is not None and pristine_keyword is not None:
                result[f"dn_{type}"] =int(type_count[type]) - next(item[f"{type}_atoms"] for item in pristine_data if pristine_keyword in str(item['folder']) )
            
        structure_data.append(result)
    df_structure = pd.DataFrame(data=structure_data)
    return df_structure, structure_data


if __name__ == "__main__":
    # Conversion constant:
    CONV_FACTOR = 16021.8 # eV/A2 to mJ/m2
    pot_prefix = "GRACE-2L-OMAT"  # PACE-NiNb-20231016, GRACE-2L-OMAT, PACE-NiNb-AUG-6Nov
    elem = ["Ta", "Fe"]  
    
    step1_folder = pathlib.Path("STEP1_UNITCELL")
    step2_folder = pathlib.Path("STEP2a_SUPERCELL_PF")
                                
    # bulk structure data defined at the upper and lower limits
    bulk_dirlist = os.listdir(step1_folder/"SIMULATIONS")
    bulk_path = [f"{step1_folder}/SIMULATIONS/{dir}" for dir in bulk_dirlist]
    df_bulk , data_bulk = collect_structure(bulk_path)

    # defect structure data 
    defect_dirlist = os.listdir(step2_folder/"SIMULATIONS")
    list_categories = ['PRISTINE', 'PYRTB', 'A6B7', 'A7B6'] # make this user-defined later
    
    # Find filenames that contain 'PRISTINE'
    basal_pristine_files = [filename for filename in defect_dirlist if 'PRISTINE' in filename and not 'PYR' in filename]
    basal_defect_A6B7_files = [filename for filename in defect_dirlist if 'PRISTINE' not in filename and 'A6B7' in filename and not 'PYR' in filename]
    basal_defect_A7B6_files = [filename for filename in defect_dirlist if 'PRISTINE' not in filename and 'A7B6' in filename and not 'PYR' in filename]
    pyr_pristine_files = [filename for filename in defect_dirlist if 'PRISTINE' in filename and 'PYR' in filename]
    pyr_defect_A6B7_files = [filename for filename in defect_dirlist if 'PRISTINE' not in filename and 'A6B7' in filename and 'PYRTB' in filename]
    pyr_defect_A7B6_files = [filename for filename in defect_dirlist if 'PRISTINE' not in filename and 'A7B6' in filename and 'PYRTB' in filename]

    # Define structure path files
    basal_pristine_path = [pathlib.Path(step2_folder/"SIMULATIONS")/dir for dir in basal_pristine_files]
    basal_defect_A6B7_path = [pathlib.Path(step2_folder/"SIMULATIONS")/dir for dir in basal_defect_A6B7_files]
    basal_defect_A7B6_path = [pathlib.Path(step2_folder/"SIMULATIONS")/dir for dir in basal_defect_A7B6_files]
    pyr_pristine_path = [pathlib.Path(step2_folder/"SIMULATIONS")/dir for dir in pyr_pristine_files]
    pyr_defect_A6B7_path = [pathlib.Path(step2_folder/"SIMULATIONS")/dir for dir in pyr_defect_A6B7_files]
    pyr_defect_A7B6_path = [pathlib.Path(step2_folder/"SIMULATIONS")/dir for dir in pyr_defect_A7B6_files]

    # Collect structure data from filepaths
    df_basal_pristine, data_basal_pristine = collect_structure(basal_pristine_path)
    df_basal_defect_A6B7, data_basal_defect_A6B7 = collect_structure(basal_defect_A6B7_path,pristine_data = data_basal_pristine, pristine_keyword = 'A6B7')
    df_basal_defect_A7B6, data_basal_defect_A7B6 = collect_structure(basal_defect_A7B6_path,pristine_data = data_basal_pristine, pristine_keyword = 'A7B6')
    df_pyr_pristine, data_pyr_pristine = collect_structure(pyr_pristine_path)
    df_pyr_defect_A6B7, data_pyr_defect_A6B7 = collect_structure(pyr_defect_A6B7_path,pristine_data = data_pyr_pristine, pristine_keyword = 'A6B7')
    df_pyr_defect_A7B6, data_pyr_defect_A7B6 = collect_structure(pyr_defect_A7B6_path,pristine_data = data_pyr_pristine, pristine_keyword = 'A7B6')


    # Define reference energies needed to calculate defect formation energy
    C14_AB2_cto_A6B7 = df_bulk.loc[df_bulk['folder'].eq("C14_AB2_cto_A6B7"),'E_cell'].values[0]
    C14_AB2_cto_A6B7_Ta_atoms = df_bulk.loc[df_bulk['folder'].eq("C14_AB2_cto_A6B7"),'Ta_atoms'].values[0]
    C14_AB2_cto_A6B7_Fe_atoms = df_bulk.loc[df_bulk['folder'].eq("C14_AB2_cto_A6B7"),'Fe_atoms'].values[0]

    A6B7_sup555 = df_bulk.loc[df_bulk['folder'].eq("A6B7_sup555"),'E_cell'].values[0]
    A6B7_sup555_Ta_atoms = df_bulk.loc[df_bulk['folder'].eq("A6B7_sup555"),'Ta_atoms'].values[0]
    A6B7_sup555_Fe_atoms = df_bulk.loc[df_bulk['folder'].eq("A6B7_sup555"),'Fe_atoms'].values[0]

    A7B6_sup555 = df_bulk.loc[df_bulk['folder'].eq("A7B6_sup555"),'E_cell'].values[0]
    A7B6_sup555_Ta_atoms = df_bulk.loc[df_bulk['folder'].eq("A7B6_sup555"),'Ta_atoms'].values[0]
    A7B6_sup555_Fe_atoms = df_bulk.loc[df_bulk['folder'].eq("A7B6_sup555"),'Fe_atoms'].values[0]

    A7B6_cto_A6B7_sup555 = df_bulk.loc[df_bulk['folder'].eq("A7B6_cto_A6B7"),'E_cell'].values[0]
    A7B6_cto_A6B7_sup555_Ta_atoms = df_bulk.loc[df_bulk['folder'].eq("A7B6_cto_A6B7"),'Ta_atoms'].values[0]
    A7B6_cto_A6B7_sup555_Fe_atoms = df_bulk.loc[df_bulk['folder'].eq("A7B6_cto_A6B7"),'Fe_atoms'].values[0]

    A6B7_cto_A7B6_sup555 = df_bulk.loc[df_bulk['folder'].eq("A6B7_cto_A7B6"),'E_cell'].values[0]
    A6B7_cto_A7B6_sup555_Ta_atoms = df_bulk.loc[df_bulk['folder'].eq("A6B7_cto_A7B6"),'Ta_atoms'].values[0]
    A6B7_cto_A7B6_sup555_Fe_atoms = df_bulk.loc[df_bulk['folder'].eq("A6B7_cto_A7B6"),'Fe_atoms'].values[0]

    A_bulk = df_bulk.loc[df_bulk['folder'].eq("A_bulk"),'E_cell'].values[0]
    A_bulk_Ta_atoms = df_bulk.loc[df_bulk['folder'].eq("A_bulk"),'Ta_atoms'].values[0]
    A_bulk_Fe_atoms = df_bulk.loc[df_bulk['folder'].eq("A_bulk"),'Fe_atoms'].values[0]


    # A6B7 PHASE DIAGRAM PLOT -----------------------------

    # Looking at TaFe2 + Ta6Fe7
    # Ta Fe
    poor_coeff = np.array([[C14_AB2_cto_A6B7_Ta_atoms,C14_AB2_cto_A6B7_Fe_atoms] , [A6B7_sup555_Ta_atoms, A6B7_sup555_Fe_atoms]   ] )
    poor_res = [C14_AB2_cto_A6B7 , A6B7_sup555 ]
    mu_poor = np.linalg.solve(poor_coeff,poor_res)
    poorTa_Ta = mu_poor[0]
    poorTa_Fe = mu_poor[1]

    # Looking at Ta6Fe7 + Ta7Fe6(cto Ta6Fe7)
    # Ta Fe
    rich_coeff = np.array([[A6B7_sup555_Ta_atoms,A6B7_sup555_Fe_atoms] , [A7B6_cto_A6B7_sup555_Ta_atoms, A7B6_cto_A6B7_sup555_Fe_atoms]  ] )
    rich_res = np.array([A6B7_sup555, A7B6_cto_A6B7_sup555])
    mu_rich =  np.linalg.solve(rich_coeff,rich_res)
    richTa_Ta = mu_rich[0]
    richTa_Fe = mu_rich[1]

    # BASAL DEFECT 
    E_cell_A7B6 = float(next(item['E_cell'] for item in data_basal_pristine if 'A7B6_PRISTINE' in str(item['folder'])))
    E_cell_A6B7 = float(next(item['E_cell'] for item in data_basal_pristine if 'A6B7_PRISTINE' in str(item['folder'])))

    for defect in data_basal_defect_A6B7: 
        defect['Ef_poorTa'] = (defect['E_cell'] - E_cell_A6B7 - defect['dn_Fe']*poorTa_Fe - defect['dn_Ta']*poorTa_Ta)*16021.8/defect['area_A2']/2
        defect['Ef_richTa'] = (defect['E_cell'] - E_cell_A6B7 - defect['dn_Fe']*richTa_Fe - defect['dn_Ta']*richTa_Ta)*16021.8/defect['area_A2']/2
        
    # PYRAMIDAL DEFECT 
    E_cell_A7B6_pyramidal = float(next(item['E_cell'] for item in data_pyr_pristine if 'A7B6_PYRPRISTINE' in str(item['folder']) ))
    E_cell_A6B7_pyramidal = float(next(item['E_cell'] for item in data_pyr_pristine if 'A6B7_PYRPRISTINE' in str(item['folder']) ))

    for defect in data_pyr_defect_A6B7: 
        defect['Ef_poorTa'] = (defect['E_cell'] - E_cell_A6B7_pyramidal - defect['dn_Fe']*poorTa_Fe - defect['dn_Ta']*poorTa_Ta)*16021.8/defect['area_A2']/2
        defect['Ef_richTa'] = (defect['E_cell'] - E_cell_A6B7_pyramidal - defect['dn_Fe']*richTa_Fe - defect['dn_Ta']*richTa_Ta)*16021.8/defect['area_A2']/2
        
    xrange = np.linspace(-0.8,0.1)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=300)

    plt.axhline(y=0, color='black', linewidth=1)

    x = [poorTa_Ta - A_bulk/A_bulk_Ta_atoms, richTa_Ta -A_bulk/A_bulk_Ta_atoms]

    for defect in data_basal_defect_A6B7:
        y = [defect['Ef_poorTa'],defect['Ef_richTa']]
        fit =  np.polyfit(x,y,1)
        line = np.poly1d(fit)
        y_extrapolated = line(xrange)
        ax.plot(xrange,y_extrapolated, label=defect['folder'])
        # ax.plot(xrange,y_extrapolated, label=defect['label'],color=defect['color'])

    for defect in data_pyr_defect_A6B7:
        y = [defect['Ef_poorTa'],defect['Ef_richTa']]
        fit =  np.polyfit(x,y,1)
        line = np.poly1d(fit)
        y_extrapolated = line(xrange)
        ax.plot(xrange,y_extrapolated, label=defect['folder'])
        # ax.plot(xrange,y_extrapolated, label=defect['label'],color=defect['color'])

    for xi in x: # Add vertical dotted lines
        plt.axvline(x=xi, color='black', linestyle='dotted', linewidth=3)



    plt.legend(loc='lower right',fontsize=10, facecolor="white",frameon=True, framealpha=1)

    plt.xlim(-0.41,0.1)
    plt.ylim(-500,500)
    plt.xlabel(r'$\Delta \mu_{A} = \mu_{A} - \mu_{A}^0 $', fontsize=14)
    plt.ylabel(r'Formation Energy, mJ/$\mathrm{m^2}$',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(r'Defect Phase Diagn : A$_6$B$_7$')
    plt.tight_layout()
    # plt.savefig('Ta6Fe7_phasediagn_supercell_cto.svg', bbox_inches="tight")
    plt.savefig('A6B7_phasediagn_supercell_CTO.png', bbox_inches="tight")



    # A7B6 PHASE DIAGRAM PLOT -----------------------------
    # Ta Fe
    poor_coeff = np.array([[A6B7_cto_A7B6_sup555_Ta_atoms,A6B7_cto_A7B6_sup555_Fe_atoms] , [A7B6_sup555_Ta_atoms, A7B6_sup555_Fe_atoms]   ] )
    poor_res = [A6B7_cto_A7B6_sup555, A7B6_sup555]
    mu_poor = np.linalg.solve(poor_coeff,poor_res)
    poorTa_Ta = mu_poor[0] # energy per Ta
    poorTa_Fe = mu_poor[1] # energy per Fe


    # Looking at Ta bulk and Ta7Fe6
    # Ta Fe
    rich_coeff = np.array([[A7B6_sup555_Ta_atoms,A7B6_sup555_Fe_atoms] , [A_bulk_Ta_atoms,0]  ] )
    rich_res = np.array([A7B6_sup555, A_bulk])
    mu_rich =  np.linalg.solve(rich_coeff,rich_res)
    richTa_Ta = mu_rich[0] # energy per Ta
    richTa_Fe = mu_rich[1] # energy per Fe


    for defect in data_basal_defect_A7B6: 
        defect['Ef_poorTa'] = (defect['E_cell'] - E_cell_A7B6 - defect['dn_Fe']*poorTa_Fe - defect['dn_Ta']*poorTa_Ta)*16021.8/defect['area_A2']/2
        defect['Ef_richTa'] = (defect['E_cell'] - E_cell_A7B6 - defect['dn_Fe']*richTa_Fe - defect['dn_Ta']*richTa_Ta)*16021.8/defect['area_A2']/2
        
    for defect in data_pyr_defect_A7B6: 
        defect['Ef_poorTa'] = (defect['E_cell'] - E_cell_A7B6_pyramidal - defect['dn_Fe']*poorTa_Fe - defect['dn_Ta']*poorTa_Ta)*16021.8/defect['area_A2']/2
        defect['Ef_richTa'] = (defect['E_cell'] - E_cell_A7B6_pyramidal - defect['dn_Fe']*richTa_Fe - defect['dn_Ta']*richTa_Ta)*16021.8/defect['area_A2']/2


    xrange = np.linspace(-9,0.1)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=300)

    plt.axhline(y=0, color='black', linewidth=1)

    x = [poorTa_Ta - A_bulk/A_bulk_Ta_atoms, richTa_Ta -A_bulk/A_bulk_Ta_atoms]
    for defect in data_basal_defect_A7B6:
        y = [defect['Ef_poorTa'],defect['Ef_richTa']]
        fit =  np.polyfit(x,y,1)
        line = np.poly1d(fit)
        y_extrapolated = line(xrange)
        ax.plot(xrange,y_extrapolated, label=defect['folder'])


    for defect in data_pyr_defect_A7B6:
        y = [defect['Ef_poorTa'],defect['Ef_richTa']]
        fit =  np.polyfit(x,y,1)
        line = np.poly1d(fit)
        y_extrapolated = line(xrange)
        ax.plot(xrange,y_extrapolated, label=defect['folder'])

    
    for xi in x:  # Add vertical dotted lines
        plt.axvline(x=xi, color='black', linestyle='dotted', linewidth=3)

    plt.legend()
    plt.xlim(-0.41,0.1)
    plt.ylim(-500,500)
    plt.xlabel(r'$\Delta \mu_{A} = \mu_{A} - \mu_{A}^0 $', fontsize=14)
    plt.ylabel(r'Formation Energy, mJ/$\mathrm{m^2}$',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(r'Defect Phase Diagn : A$_7$B$_6$')
    plt.tight_layout()
    # plt.savefig('Ta7Fe6_phasediagn_supercell_cto.svg', bbox_inches="tight")
    plt.savefig('A7B6_phasediagn_supercell_CTO.png', bbox_inches="tight")