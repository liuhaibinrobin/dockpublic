import os
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np

def sx_ligand_dedocking(mol, generate_3D_conf = False, random_seed = 10):
    if generate_3D_conf:
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=1, randomSeed=random_seed)
        mol = AllChem.RemoveHs(mol)
    else:
        mol.RemoveAllConformers()
    return mol
'''
def sx_get_nci_matrix(protein_name, ligand_name, res_full_id, atom_names, nci_df):
    matrix = np.zeros((len(res_full_id),len(atom_names)))
    try:
        atom_names = {_atom:i for (_atom, i) in zip(atom_names, range(len(atom_names)))}
        #print(atom_names)
        res_full_id = {_res:i for (_res, i) in zip(res_full_id, range(len(res_full_id)))}
        #print(res_full_id)
        nci_df = nci_df[nci_df.PDB_Code == protein_name]
        nci_df = nci_df[nci_df.LigName == ligand_name]
        #print(nci_df)
        for (_, line) in nci_df.iterrows():
            _atom_name = line["LigAtomName"]
            _res_full_id = line["ResFullID"]
            #print(_res_full_id, _atom_name)
            if _res_full_id in res_full_id.keys():
                matrix[res_full_id[_res_full_id]][atom_names[_atom_name]] = 1
        return matrix
    except Exception as e:
        print("Ligand and protein name",ligand_name, protein_name, ": error in sx_get_nci_matrix :", e.__class__.__name__,e)
        return np.zeros((len(res_full_id),len(atom_names)))
'''

def sx_get_nci_matrix_by_dict(protein_name, ligand_name, res_ids, atom_ids, nci_df):
    matrix = np.zeros((len(res_ids), len(atom_ids)))
    try:
        nci_df = nci_df[nci_df.PDB_Code == protein_name]
        nci_df = nci_df[nci_df.LigName == ligand_name]
        tmp = [i for i in range(len(res_ids))]
        res_ids = {_res:i for (_res, i) in zip(res_ids, tmp)}
        #print(nci_df)
        for (_, line) in nci_df.iterrows():
            _atom_id = line["LigAtomName"]
            _res_id = line["ResFullID"]
            if _res_id in res_ids.keys():
                matrix[res_ids[_res_id]][atom_ids[_atom_id]] = 1
        return matrix
    except Exception as e:
        # prrint("Ligand and protein name",ligand_name, protein_name, ": error in sx_get_nci_matrix_name_by_dict:", e.__class__.__name__,e)
        return np.zeros((len(res_ids),len(atom_ids)))
'''
def sx_get_nci_matrix_with_name(protein_name, ligand_name, res_full_id, atom_names, nci_df):
    matrix = np.zeros((len(res_full_id),len(atom_names)))
    try:
        atom_names = {_atom:i for (_atom, i) in zip(atom_names, range(len(atom_names)))}
        #print(atom_names)
        res_full_id = {_res:i for (_res, i) in zip(res_full_id, range(len(res_full_id)))}
        #print(res_full_id)
        nci_df = nci_df[nci_df.PDB_Code == protein_name]
        nci_df = nci_df[nci_df.LigName == ligand_name]
        #print(nci_df)
        for (_, line) in nci_df.iterrows():
            _atom_name = line["LigAtomName"]
            _res_full_id = line["ResFullID"]
            #print(_res_full_id, _atom_name)
            if _res_full_id in res_full_id.keys():
                matrix[res_full_id[_res_full_id]][atom_names[_atom_name]] = 1
        return matrix, atom_names, res_full_id
    except Exception as e:
        print("Ligand and protein name",ligand_name, protein_name, ": error in sx_get_nci_matrix :", e.__class__.__name__,e)
        return np.zeros((len(res_full_id),len(atom_names))), atom_names, res_full_id
'''