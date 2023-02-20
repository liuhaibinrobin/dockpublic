# training script for new model
# prototype 2022-12-27

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, InMemoryDataset, download_url
from utils import construct_data_from_graph_gvp
from feature_utils import extract_torchdrug_feature_from_mol, get_canonical_smiles
import rdkit.Chem as Chem



# data
#   compound_name: should be a smiles

class TankBindDataSet_prototype(Dataset):   
    def __init__(self, 
                root, 
                data=None, 
                protein_dict=None, 
                compound_dict=None, 
                proteinMode=0, 
                compoundMode=1,
                add_noise_to_com=None, 
                pocket_radius=20, 
                contactCutoff=8.0, 
                predDis=True,               
                shake_nodes=None,
                transform=None, pre_transform=None,
                 pre_filter=None,session_type=None):
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.data["session_ap"] = self.data['assay_id'].astype(str) + "_" + self.data['pdb_id']

        self.protein_dict = torch.load(self.processed_paths[1])
        self.compound_dict = torch.load(self.processed_paths[2])
        self.add_noise_to_com = add_noise_to_com
        self.proteinMode = proteinMode
        self.compoundMode = compoundMode
        self.pocket_radius = pocket_radius
        self.contactCutoff = contactCutoff
        self.predDis = predDis
        self.shake_nodes = shake_nodes
        self.session_type=session_type
        if self.session_type=="session_au":
            self.data["session"]=self.data["session_au"]
        elif  self.session_type=="session_ap":
            self.data["session"]=self.data["session_ap"]

    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt', 'compound.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])
        torch.save(self.compound_dict, self.processed_paths[2])

    def len(self):
        return len(self.data)

    def get(self, idx):
        if self.session_type==None:
            raise Exception

        line = self.data.iloc[idx]
        
        pocket_com = line['pocket_com']
        # use_compound_com = line['use_compound_com'] # 在原设定中，仅有 natvie_pocket 此项目为 TRUE，我们并没有 native_pocket，故注释掉。
        
        # use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False # 实际全为 False，故取消
        
        
        split_tag = line['split_tag']
        
        # add_noise_to_com = self.add_noise_to_com if split_tag == 'train' else None  # TODO: Need this ?

        PDB_id = line['pdb_id']
        pdb_id = PDB_id.lower()
        if self.proteinMode == 0:
            # protein embedding follow GVP protocol.
            protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[pdb_id]

        smiles_name = line['smiles_name']        
        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = self.compound_dict[smiles_name]

        # node_xyz could add noise too.
        # shake_nodes = self.shake_nodes if split_tag == 'train' else None
        # if shake_nodes is not None:
        #     protein_node_xyz = protein_node_xyz + shake_nodes * (2 * np.random.rand(*protein_node_xyz.shape) - 1)
        #     coords = coords + shake_nodes * (2 * np.random.rand(*coords.shape) - 1)

        if self.proteinMode == 0:
            data, input_node_list, keepNode = construct_data_from_graph_gvp(
                protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, contactCutoff=self.contactCutoff, includeDisMap=self.predDis,
                pocket_radius=self.pocket_radius, 
                add_noise_to_com=None, 
                use_whole_protein=False, 
                use_compound_com_as_pocket=False, 
                chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)

        # affinity = affinity_to_native_pocket * min(1, float((data.y.numpy() > 0).sum()/(5*coords.shape[0])))
        ## affinity = float(line['affinity'])
        ## data.affinity = torch.tensor([affinity], dtype=torch.float)
        
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)
        data.pdb_id = line['pdb_id']
        data.sample_id = line['sample_id']
        data.split_tag = split_tag

        data.session = line['session']

        data.value = line['value']
        ### use_coumpound_com 对于非 native pocket 均为 False
        data.real_affinity_mask = torch.tensor([False], dtype=torch.bool)
        ## data.real_y_mask = torch.ones(data.y.shape).bool() if use_compound_com else torch.zeros(data.y.shape).bool()
        data.real_y_mask = torch.zeros(data.y.shape).bool()


        return data




def get_data_prototype(pre, data_mode, addNoise=None):
         

    return train_dataset, iid_dataset, ood_dataset, test_dataset



def get_full_data_prototype(pre, data_mode, random_state=0, addNoise=None,session_type=None):
    if session_type ==None :
        raise Exception

    if data_mode not in ['small', 'full', 'duplicated']:
        raise ValueError("data_mode should be chosen from ['small' and 'full']， or `duplicated` in rare cases.")
    
    
    if data_mode == "small":
        add_noise_to_com = None
        
        train_dataset = TankBindDataSet_prototype(root=f"{pre}/small_train", add_noise_to_com=None)
        train_dataset.data = train_dataset.data.query("c_length < 100").reset_index(drop=True)
        if session_type=="session_au":
            train_dataset.data = train_dataset.data.groupby("session_aus").sample(n=1).reset_index(drop=True)

        
        iid_dataset = TankBindDataSet_prototype(root=f"{pre}/small_val", add_noise_to_com=None)
        iid_index = iid_dataset.data.query("split_tag =='iid_val'").index.values
        if len(iid_index>0):
            iid_dataset.data = iid_dataset.data.iloc[iid_index].reset_index(drop=True) 
        else:
            iid_dataset = None
        
        ood_dataset = TankBindDataSet_prototype(root=f"{pre}/small_val", add_noise_to_com=None)
        ood_index = ood_dataset.data.query("split_tag =='ood_val'").index.values
        if len(ood_index>0):
            ood_dataset.data = ood_dataset.data.iloc[ood_index].reset_index(drop=True)
        else:
            ood_dataset = None
        
        test_dataset = TankBindDataSet_prototype(root=f"{pre}/small_val", add_noise_to_com=None)
        test_index = test_dataset.data.query("split_tag =='test'").index.values
        if len(test_index>0):
            test_dataset.data = test_dataset.data.iloc[test_index].reset_index(drop=True)
        else:
            test_dataset = None 
            
    elif data_mode == "full":
        add_noise_to_com = None
        
        train_dataset = TankBindDataSet_prototype(root=f"{pre}/full", add_noise_to_com=None)
        train_dataset.data = train_dataset.data.query("split_tag =='train' and c_length < 100").reset_index(drop=True)
        if session_type == "session_au":
            train_dataset.data = train_dataset.data.groupby("session_aus").sample(n=1, random_state=random_state).reset_index(drop=True)
        
        # Replaced with 0130 reduced dataset: one session, one smiles => unique pdb
        iid_dataset = TankBindDataSet_prototype(root=f"{pre}/extra_val_test_reduced_0130", add_noise_to_com=None)
        iid_index = iid_dataset.data.query("split_tag =='iid_val'").index.values
        if len(iid_index>0):
            iid_dataset.data = iid_dataset.data.iloc[iid_index].reset_index(drop=True) 
        else:
            iid_dataset = None
        
        ood_dataset = TankBindDataSet_prototype(root=f"{pre}/extra_val_test_reduced_0130", add_noise_to_com=None)
        ood_index = ood_dataset.data.query("split_tag =='ood_val'").index.values
        if len(ood_index>0):
            ood_dataset.data = ood_dataset.data.iloc[ood_index].reset_index(drop=True)
        else:
            ood_dataset = None
        
        test_dataset = TankBindDataSet_prototype(root=f"{pre}/extra_val_test_reduced_0130", add_noise_to_com=None)
        test_index = test_dataset.data.query("split_tag =='test'").index.values
        if len(test_index>0):
            test_dataset.data = test_dataset.data.iloc[test_index].reset_index(drop=True)
        else:
            test_dataset = None 
        
    elif data_mode == "duplicated":
        add_noise_to_com = None
        
        train_dataset = TankBindDataSet_prototype(root=f"{pre}/full", add_noise_to_com=None)
        train_dataset.data = train_dataset.data.query("split_tag =='train' and c_length < 100").reset_index(drop=True)
        if session_type == "session_au":
            train_dataset.data = train_dataset.data.groupby("session_aus").sample(n=1).reset_index(drop=True)
        
        # Replaced with 0130 reduced dataset: one session, one smiles => unique pdb
        iid_dataset = TankBindDataSet_prototype(root=f"{pre}/full", add_noise_to_com=None)
        iid_index = iid_dataset.data.query("split_tag =='iid_val'").index.values
        if len(iid_index>0):
            iid_dataset.data = iid_dataset.data.iloc[iid_index].reset_index(drop=True) 
        else:
            iid_dataset = None
        
        ood_dataset = TankBindDataSet_prototype(root=f"{pre}/full", add_noise_to_com=None)
        ood_index = ood_dataset.data.query("split_tag =='ood_val'").index.values
        if len(ood_index>0):
            ood_dataset.data = ood_dataset.data.iloc[ood_index].reset_index(drop=True)
        else:
            ood_dataset = None
        
        test_dataset = TankBindDataSet_prototype(root=f"{pre}/full", add_noise_to_com=None)
        test_index = test_dataset.data.query("split_tag =='test'").index.values
        if len(test_index>0):
            test_dataset.data = test_dataset.data.iloc[test_index].reset_index(drop=True)
        else:
            test_dataset = None       
    
    return train_dataset, iid_dataset, ood_dataset, test_dataset
        
def get_internal_dataset(pre, addNoise=None):
    internal_dataset = TankBindDataSet_prototype(root=f"{pre}/internal", add_noise_to_com=None)
    return internal_dataset



# Trash
'''
# Unused ?
class TankBindDataSet_qsar(Dataset):
    def __init__(self, root, data=None, protein_dict=None, compound_dict=None, proteinMode=0, compoundMode=1,
                add_noise_to_com=None, pocket_radius=20, contactCutoff=8.0, predDis=True, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.protein_dict = torch.load(self.processed_paths[1])
        self.compound_dict = torch.load(self.processed_paths[2])
        self.add_noise_to_com = add_noise_to_com
        self.proteinMode = proteinMode
        self.compoundMode = compoundMode
        self.pocket_radius = pocket_radius
        self.contactCutoff = contactCutoff
        self.predDis = predDis
        self.shake_nodes = shake_nodes
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt', 'compound.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])
        torch.save(self.compound_dict, self.processed_paths[2])

    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        # uid = line['uid']
        smiles = line['smiles']
        pocket_com = line['pocket_com']
        use_compound_com = line['use_compound_com']
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False
        group = line['group'] if "group" in line.index else 'train'
        add_noise_to_com = self.add_noise_to_com if group == 'train' else None

        protein_name = line['protein_name']
        if self.proteinMode == 0:
            # protein embedding follow GVP protocol.
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
        

        # node_xyz could add noise too.
        shake_nodes = self.shake_nodes if group == 'train' else None
        if shake_nodes is not None:
            protein_node_xyz = protein_node_xyz + shake_nodes * (2 * np.random.rand(*protein_node_xyz.shape) - 1)
            coords = coords  + shake_nodes * (2 * np.random.rand(*coords.shape) - 1)

        if self.proteinMode == 0:
            data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                  protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                                  coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, contactCutoff=self.contactCutoff, includeDisMap=self.predDis,
                          pocket_radius=self.pocket_radius, add_noise_to_com=add_noise_to_com, use_whole_protein=use_whole_protein, 
                          use_compound_com_as_pocket=use_compound_com, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)

        # affinity = affinity_to_native_pocket * min(1, float((data.y.numpy() > 0).sum()/(5*coords.shape[0])))
        # affinity = float(line['affinity'])
        # data.affinity = torch.tensor([affinity], dtype=torch.float)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)
        data.pdb = line['pdb'] if "pdb" in line.index else f'smiles_{idx}'
        data.group = group

        data.real_affinity_mask = torch.tensor([use_compound_com], dtype=torch.bool)
        data.real_y_mask = torch.ones(data.y.shape).bool() if use_compound_com else torch.zeros(data.y.shape).bool()
        # fract_of_native_contact = float(line['fract_of_native_contact']) if "fract_of_native_contact" in line.index else 1
        # equivalent native pocket
        if "native_num_contact" in line.index:
            fract_of_native_contact = (data.y.numpy() > 0).sum() / float(line['native_num_contact'])
            is_equivalent_native_pocket = fract_of_native_contact >= 0.9
            data.is_equivalent_native_pocket = torch.tensor([is_equivalent_native_pocket], dtype=torch.bool)
            data.equivalent_native_y_mask = torch.ones(data.y.shape).bool() if is_equivalent_native_pocket else torch.zeros(data.y.shape).bool()
        else:
            # native_num_contact information is not available.
            # use ligand com to determine if this pocket is equivalent to native pocket.
            if "ligand_com" in line.index:
                ligand_com = line["ligand_com"]
                pocket_com = data.node_xyz.numpy().mean(axis=0)
                dis = np.sqrt(((ligand_com - pocket_com)**2).sum())
                # is equivalent native pocket if ligand com is less than 8 A from pocket com.
                is_equivalent_native_pocket = dis < 8
                data.is_equivalent_native_pocket = torch.tensor([is_equivalent_native_pocket], dtype=torch.bool)
                data.equivalent_native_y_mask = torch.ones(data.y.shape).bool() if is_equivalent_native_pocket else torch.zeros(data.y.shape).bool()
            else:
                # data.is_equivalent_native_pocket and data.equivalent_native_y_mask will not be available.
                pass
        return data
        
class TankBind_prediction(Dataset):
    def __init__(self, root, data=None, protein_dict=None, compound_dict=None, proteinMode=0, compoundMode=1,
                pocket_radius=20, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.protein_dict = torch.load(self.processed_paths[1])
        self.compound_dict = torch.load(self.processed_paths[2])
        self.proteinMode = proteinMode
        self.pocket_radius = pocket_radius
        self.compoundMode = compoundMode
        self.shake_nodes = shake_nodes
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt', 'compound.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])
        torch.save(self.compound_dict, self.processed_paths[2])

    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        pocket_com = line['pocket_com']
        pocket_com = np.array(pocket_com.split(",")).astype(float) if type(pocket_com) == str else pocket_com
        pocket_com = pocket_com.reshape((1, 3))
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False

        protein_name = line['protein_name']
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[protein_name]

        compound_name = line['compound_name']
        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = self.compound_dict[compound_name]

        # y is distance map, instead of contact map.
        data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                              protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                              coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                              pocket_radius=self.pocket_radius, use_whole_protein=use_whole_protein, includeDisMap=True,
                              use_compound_com_as_pocket=False, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)

        return data


'''