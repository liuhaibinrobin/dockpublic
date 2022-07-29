tankbind_src_folder_path = "../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)


from Bio.PDB.PDBList import PDBList   # pip install biopython if import failure
import os
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from IPython import embed
import torch
torch.set_num_threads(1)
from torch_geometric.data import Dataset
from utils import construct_data_from_graph_gvp
import rdkit.Chem as Chem    # conda install rdkit -c rdkit if import failure.
from feature_utils import extract_torchdrug_feature_from_mol, get_canonical_smiles
from feature_utils import get_protein_feature
from Bio.PDB import PDBParser
from feature_utils import get_clean_res_list
import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm    # pip install tqdm if fails.
from model import get_model

pdb_name = "6scm"
cp_name = "SOS1"
csv_name = "SOS1-ID-SMILES.csv"
# true_pocket = 4
# pocket_num = 9
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

base_pre = f"./NCIVS"
pre = f"{base_pre}/{cp_name}-{pdb_name}-07200756"
# pre = f"{base_pre}/{cp_name}-{pdb_name}-test"
os.system(f"mkdir -p {pre}")
os.system(f"rm -rf {pre}/sdfs")
os.system(f"mkdir -p {pre}/sdfs")
os.system(f"rm -rf {pre}/PDBs")
os.system(f"mkdir -p {pre}/PDBs")
os.system(f"rm -rf {pre}/p2rank")
os.system(f"mkdir -p {pre}/p2rank")
proteinName = pdb_name
proteinFile = f"{pre}/{proteinName}.pdb"

############################
# precess data
parser = PDBParser(QUIET=True)
protein_dict = {}
proteinName = pdb_name
proteinFile = f"{pre}/{proteinName}.pdb"
s = parser.get_structure("example", proteinFile)
res_list = list(s.get_residues())
clean_res_list = get_clean_res_list(res_list, ensure_ca_exist=True)
protein_dict[proteinName] = get_protein_feature(clean_res_list)
ds = f"{pre}/protein_list.ds"
with open(ds, "w") as out:
    out.write(f"/{proteinName}.pdb\n")
p2rank = "bash ./p2rank_2.3/prank"
cmd = f"{p2rank} predict {ds} -o {pre}/p2rank -threads 1"
os.system(cmd)
d = pd.read_csv(f"{pre}/{csv_name}")
info = []
for i, line in tqdm(d.iterrows(), total=d.shape[0]):
    smiles = line['smiles']
    compound_name = ""
    protein_name = proteinName
    if True:
        p2rankFile = f"{pre}/p2rank/{proteinName}.pdb_predictions.csv"
        pocket = pd.read_csv(p2rankFile)
        pocket.columns = pocket.columns.str.strip()
        pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
        for ith_pocket, com in enumerate(pocket_coms):
            com = ",".join([str(a.round(3)) for a in com])
            info.append([protein_name, compound_name, smiles, f"pocket_{ith_pocket+1}", com])
info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'smiles', 'pocket_name', 'pocket_com'])

##获得全部口袋数
pocket_num = len(info['pocket_name'].unique())
print("total pocket num is :", pocket_num)
class MyDataset_VS(Dataset):
    def __init__(self, root, data=None, protein_dict=None, proteinMode=0, compoundMode=1,
                pocket_radius=20, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.protein_dict = protein_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.protein_dict = torch.load(self.processed_paths[1])
        self.proteinMode = proteinMode
        self.pocket_radius = pocket_radius
        self.compoundMode = compoundMode
        self.shake_nodes = shake_nodes
        #self.printflag = True
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])

    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        smiles = line['smiles']
        pocket_com = line['pocket_com']
        pocket_com = np.array(pocket_com.split(",")).astype(float) if type(pocket_com) == str else pocket_com
        pocket_com = pocket_com.reshape((1, 3))
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False

        protein_name = line['protein_name']
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[protein_name]
        try:
            smiles = get_canonical_smiles(smiles)
            mol = Chem.MolFromSmiles(smiles)
            mol.Compute2DCoords()
                
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
        except:
            print("something wrong with ", smiles, "to prevent this stops our screening, we repalce it with a placeholder smiles 'CCC'")
            smiles = 'CCC'
            mol = Chem.MolFromSmiles(smiles)
            mol.Compute2DCoords()
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
        # y is distance map, instead of contact map.
        data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                              protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                              coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                              pocket_radius=self.pocket_radius, use_whole_protein=use_whole_protein, includeDisMap=True,
                              use_compound_com_as_pocket=False, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)
        return data
dataset_path = f"{pre}/dataset/"
os.system(f"rm -r {dataset_path}")
os.system(f"mkdir -p {dataset_path}")
dataset = MyDataset_VS(dataset_path, data=info, protein_dict=protein_dict)

batch_size = pocket_num  ##与口袋数相关
logging.basicConfig(level=logging.INFO)
model = get_model(0, logging, device)
# modelFile = "../saved_models/re_dock.pt"
# self-dock model
modelFile = "../saved_models/self_dock.pt"

model.load_state_dict(torch.load(modelFile, map_location=device))
_ = model.eval()

data_loader = DataLoader(dataset, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=8)
sum_embed_t = []
attn_embed_t = []
affinity_pred_list = []
y_pred_list = []
for data in tqdm(data_loader):
    data = data.to(device)
    y_pred, affinity_pred, sum_embed, attn_embed = model(data)
    affinity_pred_list.append(affinity_pred.detach().cpu())
    sum_embed_t.append(sum_embed.detach().cpu())
    attn_embed_t.append(attn_embed.detach().cpu())
affinity_pred_list = torch.cat(affinity_pred_list)

##获得smiles的TB最优pocket列表 -> true_pocket
info = dataset.data
info['affinity'] = affinity_pred_list
chosen = info.loc[info.groupby(['protein_name', 'smiles'],sort=False)['affinity'].agg('idxmax')].reset_index()
pocket_t = chosen['pocket_name']
true_pocket = []
for i in range(len(pocket_t)):
    true_pocket.append(int(pocket_t[i][-1]))
print("Done loading optimal pocket for each compounds")
##获得smiles在最优pocket下的embedding和z矩阵
n = 0
sum_embed_list = []
z_list = []
for data in tqdm(data_loader):
    sum_embed_list.append(sum_embed_t[n][true_pocket[n] - 1])  ##第n个口袋是TB最优口袋(索引-1)
    z_list.append(attn_embed_t[n][true_pocket[n] - 1])
    n += 1
if len(sum_embed_list) != d.shape[0]:
        raise ValueError("the number of smiles and embeddings don't match")
data = {}
for i, line in tqdm(d.iterrows(), total=d.shape[0]):
    smiles = line['smiles']
    data[smiles] = sum_embed_list[i]
with open(f'embedding/{cp_name}/{cp_name}_z_mask_dic.pickle', 'wb') as f:
    pickle.dump(data, f)

data1 = {}
for i, line in tqdm(d.iterrows(), total=d.shape[0]):
    smiles = line['smiles']
    data1[smiles] = z_list[i]
with open(f'embedding/{cp_name}/{cp_name}_z_dic.pickle', 'wb') as f:
    pickle.dump(data1, f)

print(f'Done preprocessing embedding for {cp_name}')