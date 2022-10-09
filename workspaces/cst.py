# %% [markdown]
# # overview
# 
# We start from the raw PDBbind dataset downloaded from http://www.pdbbind.org.cn/download.php
# 
# 1. filter out those unable to process using RDKit.
# 
# 2. Process the protein by only preserving the chains that with at least one atom within 10Å from any atom of the ligand.
# 
# 3. Use p2rank to segment protein into blocks.
# 
# 4. extract protein and ligand features.
# 
# 5. construct the training and test dataset.
# 

# %%
tankbind_src_folder_path = "../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)

# %%
import pandas as pd
import numpy as np
from tqdm import tqdm

# %% [markdown]
# # process the raw PDBbind dataset.

# %%
from utils import read_pdbbind_data

# %%
# raw PDBbind dataset could be downloaded from http://www.pdbbind.org.cn/download.php
pre = "/home/jovyan/data/pdbbind2020"
df_pdb_id = pd.read_csv(f'{pre}/index/INDEX_general_PL_name.2020', sep="  ", comment='#', header=None, names=['pdb', 'year', 'uid', 'd', 'e','f','g','h','i','j','k','l','m','n','o'], engine='python')
df_pdb_id = df_pdb_id[['pdb','uid']]
data = read_pdbbind_data(f'{pre}/index/INDEX_general_PL_data.2020')
data = data.merge(df_pdb_id, on=['pdb'])


# %% [markdown]
# # ligand file should be readable by RDKit.

# %%
from feature_utils import read_mol

# %%
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
pdb_list = []
probem_list = []
for pdb in tqdm(data.pdb):
    sdf_fileName = f"{pre}/pdbbind_files/{pdb}/{pdb}_ligand.sdf"
    mol2_fileName = f"{pre}/pdbbind_files/{pdb}/{pdb}_ligand.mol2"
    mol, problem = read_mol(sdf_fileName, mol2_fileName)
    if problem:
        probem_list.append(pdb)
        continue
    pdb_list.append(pdb)

# %%
data = data.query("pdb in @pdb_list").reset_index(drop=True)

# %%
data.shape

# %% [markdown]
# ### for ease of RMSD evaluation later, we renumber the atom index to be consistent with the smiles

# %%
from feature_utils import write_renumbered_sdf
pre_main = "/home/jovyan/dataspace/NFT/main"
toFolder = f"{pre_main}/renumber_atom_index_same_as_smiles"
os.system(f"mkdir -p {toFolder}")

# %%
for pdb in tqdm(pdb_list):
    sdf_fileName = f"{pre}/pdbbind_files/{pdb}/{pdb}_ligand.sdf"
    mol2_fileName = f"{pre}/pdbbind_files/{pdb}/{pdb}_ligand.mol2"
    toFile = f"{toFolder}/{pdb}.sdf"
    write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName)


# %% [markdown]
# # process PDBbind proteins, removing extra chains, cutoff 10A

# %%
toFolder = f"{pre_main}/protein_remove_extra_chains_10A/"
os.system(f"mkdir -p {toFolder}")

# %%
input_ = []
cutoff = 10
for pdb in data.pdb.values:
    pdbFile = f"{pre}/pdbbind_files/{pdb}/{pdb}_protein.pdb"
    ligandFile = f"{pre_main}/renumber_atom_index_same_as_smiles/{pdb}.sdf"
    toFile = f"{toFolder}/{pdb}_protein.pdb"
    x = (pdbFile, ligandFile, cutoff, toFile)
    input_.append(x)

# %%
from feature_utils import select_chain_within_cutoff_to_ligand_v2

# %%
import mlcrate as mlc
import os
pool = mlc.SuperPool(64)
pool.pool.restart()
_ = pool.map(select_chain_within_cutoff_to_ligand_v2,input_)
pool.exit()

# %%
# previously, I found that 2r1w has no chain near the ligand.
data = data.query("pdb != '2r1w'").reset_index(drop=True)

# %% [markdown]
# # p2rank segmentation

# %%
p2rank_prediction_folder = f"{pre_main}/p2rank_protein_remove_extra_chains_10A"
os.system(f"mkdir -p {p2rank_prediction_folder}")
ds = f"{p2rank_prediction_folder}/protein_list.ds"
with open(ds, "w") as out:
    for pdb in data.pdb.values:
        out.write(f"../protein_remove_extra_chains_10A/{pdb}_protein.pdb\n")

# %%
# takes about 30 minutes.
p2rank = "bash /home/jovyan/p2rank_2.3/prank"
cmd = f"{p2rank} predict {ds} -o {p2rank_prediction_folder}/p2rank -threads 16"
os.system(cmd)


