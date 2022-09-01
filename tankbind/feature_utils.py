from Bio.PDB import PDBParser
import pandas as pd
import numpy as np
import os
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm
import glob
import torch
import torch.nn.functional as F
from io import StringIO
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
import scipy
import scipy.spatial
import requests
from rdkit.Geometry import Point3D
import random

from torchdrug import data as td     # conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg if fail to import

def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    Chem.WrapLogs()
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem


def write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName):
    # read in mol
    mol, _ = read_mol(sdf_fileName, mol2_fileName)
    # reorder the mol atom number as in smiles.
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    w = Chem.SDWriter(toFile)
    w.write(mol)
    w.close()

def get_canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def generate_rdkit_conformation_v2(smiles, n_repeat=50):
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.RemoveAllHs(mol)
    # mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    # rid = AllChem.EmbedMolecule(mol, ps)
    for repeat in range(n_repeat):
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == 0:
            break
    if rid == -1:
        print("rid", pdb, rid)
        ps.useRandomCoords = True
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == -1:
            mol.Compute2DCoords()
        else:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # mol = Chem.RemoveAllHs(mol)
    return mol


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

#adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat

def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=None):
    pair_dis = scipy.spatial.distance.cdist(coords, coords)
    bin_size=1
    bin_min=-0.5
    bin_max=15
    if LAS_distance_constraint_mask is not None:
        pair_dis[LAS_distance_constraint_mask==0] = bin_max
        # diagonal is zero.
        for i in range(pair_dis.shape[0]):
            pair_dis[i, i] = 0
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    pair_dis_distribution = pair_dis_one_hot.float()
    return pair_dis_distribution


def extract_torchdrug_feature_from_mol(mol, has_LAS_mask=False):
    coords = mol.GetConformer().GetPositions()
    if has_LAS_mask:
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
    else:
        LAS_distance_constraint_mask = None
    pair_dis_distribution = get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=LAS_distance_constraint_mask)
    molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol),node_feature='property_prediction')
    # molstd = td.Molecule.from_molecule(mol ,node_feature=['property_prediction'])
    compound_node_features = molstd.node_feature # nodes_chemical_features
    edge_list = molstd.edge_list # [num_edge, 3]
    edge_weight = molstd.edge_weight # [num_edge, 1]
    assert edge_weight.max() == 1
    assert edge_weight.min() == 1
    assert coords.shape[0] == compound_node_features.shape[0]
    edge_feature = molstd.edge_feature # [num_edge, edge_feature_dim]
    x = (coords, compound_node_features, edge_list, edge_feature, pair_dis_distribution)
    return x

import gvp
import gvp.data

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

############new
#the dict of sidec selected by halogen or the furthest
sidec_dict = {
    'ALA': ['N'], 
    'CYS': ['SG'], 
    'ASP': ['OD2'], 
    'GLU': ['OE2'], 
    'PHE': ['CE1','CE2','CD1','CD2','CG','CZ'], 
    'GLY': ['N'], 
    'HIS': [['NE2'],['ND1']],   
    'ILE': ['CD1'], 
    'LYS': ['NZ'], 
    'LEU': ['CG'], 
    'MET': ['SD'], 
    'ASN': [['OD1'],['ND2']], 
    'PRO': ['N'], 
    'GLN': [['OE1'],['NE2']], 
    'ARG': ['CZ'], 
    'SER': ['OG'], 
    'THR': ['OG1'], 
    'VAL': ['CB'], 
    'TRP': [['NE1'], ['CE2','CD2','CE3','CZ3','CH2','CZ2']], 
    'TYR': [['OH'], ['CE1','CE2','CD1','CD2','CG','CZ']]
}

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}

#alphafold分出来的4类片段
chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
}

## define x1,x2,x3,x4
def dihedral_angle(a, b, c, d):
    """return the dihedral angle of plan abc and bcd"""
    v1 = a - b
    v2 = b - c
    v3 = d - c

    c1 = np.cross(v1, v2)
    c2 = np.cross(v3, v2)
    c3 = np.cross(c2, c1)

    v2_mag = np.linalg.norm(v2)
    return np.arctan2(np.dot(c3, v2), v2_mag * np.dot(c1, c2))

def sidechain_angle(res_list, chi_angles_atoms):
    """
    return 
    1. list of sidechain angles chi1,2,3,4, shape=[len(res_list, 4)]
    2. the ids of res which is not complete
    """
    double_res_list = ['HIS', 'ASN', 'GLN', 'TRP', 'TYR']
    dihedral_angle_list_t = []
    broken_res_id = []
    for res in res_list:
        dihedral_angle_list = []
        chi_angles_list = chi_angles_atoms[res.resname]
        chi_angles_list_len = len(chi_angles_list)
        if chi_angles_list_len != 0:
            for group in chi_angles_list:
                if (group[0] not in res) or (group[1] not in res) or (group[2] not in res) or (group[3] not in res):
                    # print(res.get_full_id()[3][1])
                    broken_res_id.append(res.get_full_id()[3][1])
                    continue
                angle_x = dihedral_angle(res[group[0]].coord, res[group[1]].coord,\
                                                res[group[2]].coord, res[group[3]].coord)
                dihedral_angle_list.append(angle_x)
        if len(dihedral_angle_list) != 4:
            for i in range(4 - len(dihedral_angle_list)):
                dihedral_angle_list.append(0)
        if res.get_full_id()[3][1] not in broken_res_id:
            if res.resname in double_res_list:
                dihedral_angle_list_t.append(dihedral_angle_list) # double res according to the sidec num
                dihedral_angle_list_t.append(dihedral_angle_list)
            else:
                dihedral_angle_list_t.append(dihedral_angle_list) 
    broken_res_id = list(set(broken_res_id))
    broken_res_id.sort()
    return dihedral_angle_list_t, broken_res_id

def clean_broken_res(res_list, broken_res_id):
    """remove broken res from broken_res_id and double res according to the sidec num"""
    clean_broken_res = []
    clean_broken_res_double = []
    double_res_list = ['HIS', 'ASN', 'GLN', 'TRP', 'TYR']
    for res in res_list:
        if res.get_full_id()[3][1] not in broken_res_id:
            clean_broken_res.append(res)
            if res.resname in double_res_list:
                clean_broken_res_double.append(res)
                clean_broken_res_double.append(res)
            else:
                clean_broken_res_double.append(res)
    return clean_broken_res, clean_broken_res_double

def return_coord(res, res_sidec_list):
    """
    return the coordinates of sidec
    if one atom, then return coord
    if 6 atoms, then it's a Aromatic ring, return the mean coord
    """
    if len(res_sidec_list) == 1:
        return res[res_sidec_list[0]].coord
    elif len(res_sidec_list) == 6:
        sidec_coord = [res[res_sidec_list[i]].coord for i in range(6)]
        return np.mean(np.concatenate(sidec_coord).reshape(-1,3), axis=0)

def sidec_coord(res_list, sidec_dict):
    """define the pos of sidec according to sidec_dict"""
    sidec_coords = []
    for res in res_list:
        res_sidec_list = sidec_dict[res.resname]
        if len(res_sidec_list) == 2:
            _sidec_coord_1 = return_coord(res, res_sidec_list[0])
            _sidec_coord_2 = return_coord(res, res_sidec_list[1])
            sidec_coords.append(_sidec_coord_1)
            sidec_coords.append(_sidec_coord_2)
        else:
            sidec_coords.append(return_coord(res, res_sidec_list))
    return sidec_coords


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = gvp.data.ProteinGraphDataset([structure])
    protein = dataset[0]
    x = (protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v)
    return x

def get_protein_feature_qsar(res_list):
    '''adding dihedral_angle chi1,2,3,4 and sidec_coord'''
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    dihedral_angle_list_t, broken_res_id = sidechain_angle(res_list, chi_angles_atoms)
    clean_broken_res_l, clean_broken_res_double = clean_broken_res(res_list, broken_res_id)
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in clean_broken_res_double]) # with double
    coords = []
    for res in clean_broken_res_double:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords # with double
    #####new
    structure['dihedral_angle'] = dihedral_angle_list_t
    structure['sidec_coord'] = sidec_coord(clean_broken_res_l, sidec_dict)
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    print(len(structure['coords']))
    print(len(structure['seq']))
    print(len(structure['dihedral_angle']))
    print(len(structure['sidec_coord']))
    dataset = gvp.data.ProteinGraphDataset_qsar([structure])
    protein = dataset[0]
    x = (protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v)
    return x 
# Seed_everything(seed=42)

# used for testing.
def remove_hetero_and_extract_ligand(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    # get all regular protein residues. and ligand.
    clean_res_list = []
    ligand_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if (not ensure_ca_exist) or ('CA' in res):
                # in rare case, CA is not exists.
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        elif hetero == 'W':
            # is water, skipped.
            continue
        else:
            ligand_list.append(res)
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list, ligand_list

def get_res_unique_id(residue):
    pdb, _, chain, (_, resid, insertion) = residue.full_id
    unique_id = f"{chain}_{resid}_{insertion}"
    return unique_id

def save_cleaned_protein(c, proteinFile):
    res_list = list(c.get_residues())
    clean_res_list, ligand_list = remove_hetero_and_extract_ligand(res_list)
    res_id_list = set([get_res_unique_id(residue) for residue in clean_res_list])

    io=PDBIO()
    class MySelect(Select):
        def accept_residue(self, residue, res_id_list=res_id_list):
            if get_res_unique_id(residue) in res_id_list:
                return True
            else:
                return False
    io.set_structure(c)
    io.save(proteinFile, MySelect())
    return clean_res_list, ligand_list

def split_protein_and_ligand(c, pdb, ligand_seq_id, proteinFile, ligandFile):
    clean_res_list, ligand_list = save_cleaned_protein(c, proteinFile)
    chain = c.id
    # should take a look of this ligand_list to ensure we choose the right ligand.
    seq_id = ligand_seq_id
    # download the ligand in sdf format from rcsb.org. because we pdb format doesn't contain bond information.
    # you could also use openbabel to do this.
    url = f"https://models.rcsb.org/v1/{pdb}/ligand?auth_asym_id={chain}&auth_seq_id={seq_id}&encoding=sdf&filename=ligand.sdf"
    r = requests.get(url)
    open(ligandFile , 'wb').write(r.content)
    return clean_res_list, ligand_list

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500, confId=0)
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol

def write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()

def generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=30, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)

def select_chain_within_cutoff_to_ligand_v2(x):
    # pdbFile = f"/pdbbind2020/pdbbind_files/{pdb}/{pdb}_protein.pdb"
    # ligandFile = f"/pdbbind2020/renumber_atom_index_same_as_smiles/{pdb}.sdf"
    # toFile = f"{toFolder}/{pdb}_protein.pdb"
    # cutoff = 10
    pdbFile, ligandFile, cutoff, toFile = x
    
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", pdbFile)
    all_res = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    all_atoms = [atom for res in all_res for atom in res.get_atoms()]
    protein_coords = np.array([atom.coord for atom in all_atoms])
    chains = np.array([atom.full_id[2] for atom in all_atoms])

    mol = Chem.MolFromMolFile(ligandFile)
    lig_coords = mol.GetConformer().GetPositions()

    protein_atom_to_lig_atom_dis = scipy.spatial.distance.cdist(protein_coords, lig_coords)

    is_in_contact = (protein_atom_to_lig_atom_dis < cutoff).max(axis=1)
    chains_in_contact = set(chains[is_in_contact])
    
    # save protein chains that belong to chains_in_contact
    class MySelect(Select):
        def accept_residue(self, residue, chains_in_contact=chains_in_contact):
            pdb, _, chain, (_, resid, insertion) = residue.full_id
            if chain in chains_in_contact:
                return True
            else:
                return False

    io=PDBIO()
    io.set_structure(s)
    io.save(toFile, MySelect())