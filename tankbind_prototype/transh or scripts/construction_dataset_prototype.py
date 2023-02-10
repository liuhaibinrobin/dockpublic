import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch

pdbbind_path = "/home/jovyan/dataspace/PDBbind2020/all_pdbbind"
proto_path = "/home/jovyan/dataspace/Prototype"
p2rank = "bash /home/jovyan/TankBind/p2rank_2.3/prank"
tankbind_src_folder_path = "../tankbind_prototype/"
import sys
sys.path.insert(0, tankbind_src_folder_path)
tankbind_data_path = f"/home/jovyan/dataspace/Prototype/tankbind_data"

print("Loading protein_dict....")

protein_dict = torch.load("/home/jovyan/dataspace/Prototype/raw_protein_dict.pt")

print("Loading compound_dict....")
compound_dict = torch.load("/home/jovyan/dataspace/Prototype/tankbind_data/compound_torchdrug_features.pt")


print("Loading info....")
info = pd.read_csv("/home/jovyan/dataspace/Prototype/tables/info_core_pockets_multiton.csv", index_col=0)
info.reset_index(inplace=True, drop=True)
info.pocket_com = info.pocket_com.apply(lambda x: np.array([float(_) for _ in x.replace("[[", "").replace("]]", "").replace("[", "").replace("]", "").split()]))
from data_prototype import TankBindDataSet_prototype

toFileFull = f"/home/jovyan/main_tankbind/dataset_prototype/full/dataset"
os.system(f"rm -rf {toFileFull}")
os.system(f"mkdir -p {toFileFull}")

print("Creating dataset....")
dataset = TankBindDataSet_prototype(toFileFull, data=info, protein_dict=protein_dict, compound_dict=compound_dict)
t = []
t_dict = {}
data = dataset.data

for i, line in tqdm(data.iterrows(), total=data.shape[0]):
    d = dataset[i]
    sample_id = line['sample_id']
    p_length = d['node_xyz'].shape[0]
    c_length = d['coords'].shape[0]
    y_length = d['y'].shape[0]
    t.append([i, sample_id, p_length, c_length, y_length])
    t_dict[sample_id] = [i, sample_id, p_length, c_length, y_length]

print("Saving t....")
torch.save(t, "/home/jovyan/dataspace/Prototype/tables/full_t.pt")
torch.save(t_dict, "/home/jovyan/dataspace/Prototype/tables/full_t_dict.pt")

t = pd.DataFrame(t, columns=['index', 'sample_id', 'p_length', 'c_length', 'y_length'])
t.to_csv("/home/jovyan/dataspace/Prototype/tables/full_t.csv")
print("Saving data....")
data = pd.concat([data, t[['p_length', 'c_length', 'y_length']]], axis=1)
torch.save(data, f"{toFileFull}/processed/data.pt")