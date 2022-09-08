
# # sx.nciyes.main.<font color=red>new</font>.ipynb
# 
# Notebook for training and inference with updated NCI data and frag data.
# 
# 


import sys
import pandas as pd
from rdkit import Chem
from Bio.PDB import PDBParser
import re
sys.path.insert(0, "../tankbind/")
import time
import torch 
import random
import numpy as np
import os
import pickle
# ## Configuration and Initilization
# ### Configuration


# You may want to change settings below.
def main(args):
    import torch
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cfg_split_strategy = (0.8,0.1,0.1)
    if args.first_run == False:
        cfg_use_saved_files = True # If you have already generated all the data required.
        cfg_save_files = False # If you want to save generated data.
        cfg_checkdata2 = False
    else:
        cfg_use_saved_files = False # If you have already generated all the data required.
        cfg_save_files = True # If you want to save generated data.
        cfg_checkdata2 = True


    cfg_mode = args.config_mode
    cfg_data_version = args.data_version


    cfg_timesplit = True # If timesplit is applicated for splitting the dataset.

    cfg_custom_dir_name = "Demo" # Prefix in the name of the output folder.
    cfg_train = True # If you want to train the model.
    cfg_distinguish_by_timestamp = True # If true, a timestamp is added to your output dir name.


    # ### Other Configuration and Initilization
    # Under normal condition, you needn't modify this chapiter.


    # Under normal condition, you should not modify the codes below.
    input_path = f"./Inputs/{cfg_mode}/{cfg_data_version}/"
    output_path = f"./Outputs/{cfg_mode}/"
    save_files_path = f"./Inputs/Savedfiles/{cfg_mode}.{cfg_data_version}/"
    p2rank_save_path = f"./Inputs/Savedfiles/p2rank/"
    p2rank_path = "../p2rank_2.3/prank"
    ds_path = "../../../" # Related path used for ds files.
    save_model_path = f"./Inputs/Savedfiles/{cfg_mode}.model/" #TODO: write your paths here or just keep this.
    pdb_df_fname = f"Data.{cfg_data_version}.PDBs.csv"
    ligand_df_fname = f"Data.{cfg_data_version}.Ligands.csv"
    pdb_df_fpath = f"{input_path}{pdb_df_fname}"
    ligand_df_fpath = f"{input_path}{ligand_df_fname}"
    datainfo = f"{input_path}Datainfo.txt"
    p2rank = f"bash {p2rank_path}"

    if cfg_mode == "nciyes":
        nci_fname = f"Data.{cfg_data_version}.NCIs.csv"
        nci_df_fpath = f"{input_path}{nci_fname}"
        
    # Under normal condition, you should not modify the codes below.
    cfg_jupyter = True # If you're using jupyter notebook. # old
    cfg_right_pocket_by_minor_distance = True # If the right pocket is chosen by calculating distance between ligand center and pocket center.

    if cfg_mode == "tankbind":
        from feature_utils import get_clean_res_list, get_protein_feature
        from feature_utils import get_canonical_smiles
    if cfg_mode == "nciyes":
        from sx_feature_utils import sx_get_protein_feature, get_clean_res_list
        from feature_utils import get_canonical_smiles
    elif cfg_mode == "frag":
        from feature_utils import get_clean_res_list, get_protein_feature_qsar
        from feature_utils import get_canonical_smiles

    if cfg_jupyter:
        from tqdm.notebook import tqdm
        R, B, S = "\033[1;31m", "\033[1;34m", "\033[0m" 
    else:
        from tqdm import tqdm
        R, B, S = "", "", "" 

    _dirname = ""
    _dirname = (_dirname + cfg_custom_dir_name) if cfg_custom_dir_name else (time.strftime("%y-%m-%d_%H%M"))
    _dirname = (_dirname + "_" + time.strftime("%y-%m-%d_%H%M")) if (cfg_custom_dir_name and cfg_distinguish_by_timestamp) else _dirname
    main_path = f"{output_path}{_dirname}/"
    log_fpath = f"{main_path}log.txt"
    if os.path.exists(log_fpath):
        os.system(f"rm -r {log_fpath}")
    if os.path.exists(main_path):
        os.system(f"rm -r {main_path}")
    for _path in [save_files_path, save_model_path, main_path]:
        os.system(f"mkdir -p {_path}")
        
    def qrint(target, jupyter= cfg_jupyter, log=log_fpath, R=R, B=B, S=S, r=True, newline=True):
        end = "\n" if newline else " "
        if r:
            print(target)
        if jupyter:
            target = target.replace(R,"").replace(B,"").replace(S,"")
        with open(log_fpath, "a") as f:
            if target != "\n":
                f.write(time.strftime("[%m-%d %H:%M:%S] ")+target.replace("\n", "                 \n")+end)
            else:
                f.write("\n")

    if cfg_mode == "nci_yes":
        def get_full_id_old(full_id_ls: list, resname):
            chain_id = full_id_ls[2]
            res_id = full_id_ls[3][1]
            return chain_id + "_" + str(res_id)+"_"+resname

    def saveconfig(cfg_use_saved_files, cfg_save_files, B=B, R=R, S=S):
        qrint(f"{R}cfg_use_saved_files{S}={B}{cfg_use_saved_files}{S}, {R}cfg_save_files{S}={B}{cfg_save_files}{S}.")
        if cfg_use_saved_files and not cfg_save_files:
            return(f"{B}Skip{S} codes and {B}use saved files{S}.")
        elif not cfg_use_saved_files and cfg_save_files:
            return(f"{B}Run{S} codes and {B}save{S} results.")
        elif not cfg_use_saved_files and not cfg_save_files:
            return(f"{B}Run{S} codes but results {B}won't be saved{S}.")
        else:
            return(f"{R}CONFIG WARNING{R}: {B}Skip{S} codes and {B}use saved files{S}.")
            
    qrint(f"Results will be saved to {B}{main_path}{S}.\n")


        
    pdb_df = pd.read_csv(pdb_df_fpath, index_col=0)
    ligand_df = pd.read_csv(ligand_df_fpath, index_col=0)
    pdb_code_list = list(pdb_df["pdb_code"])
    pdb_fpath_list = list(pdb_df["pdb_fpath"])
    qrint(f"Loaded {R}pdb_df{S} from {B}{pdb_df_fpath}{S} with length {len(pdb_df)}")
    qrint(f"Loaded {R}ligand_df{S} from {B}{ligand_df_fpath}{S} with length {len(ligand_df)}")
    if cfg_mode == "nciyes":
        nci_df = pd.read_csv(nci_df_fpath, index_col=0)
        qrint(f"Loaded {R}nci_df{S} from {B}{nci_df_fpath}{S} with length {len(nci_df)}")


    # ## Running!


    # ### Get protein features: <font color="red">protein_dict</font> (& <font color="red">protein_res_id_dict</font>)


    #time
    qrint("\n", r=False)
    qrint(f"{R}protein_dict{S} and {R}protein_res_id_dict{S}")
    qrint(saveconfig(cfg_use_saved_files, cfg_save_files))

    protein_dict = {}
    protein_res_id_dict = {} if cfg_mode=="nciyes" else None

    if not cfg_use_saved_files:
        qrint(f"Processing {R}protein_dict{S} and {R}protein_res_id_dict{S}:" if (cfg_mode == "nciyes") else f"Processing {R}protein_dict{S}")
        parser = PDBParser(QUIET=True)
        protein_dicts = [{} for n in range(11)] # 0,1,2,3,4,5,6,7,8,9,x
        #protein_res_id_dict = [{} for n in range(11)]  #0,1,2,3,4,5,6,7,8,9,x
        
        for i, (_pname, _fpath) in tqdm(enumerate(zip(pdb_code_list, pdb_fpath_list)), total=len(pdb_fpath_list)):
            s = parser.get_structure(_pname, _fpath)
            res_list = list(s.get_residues())
            clean_res_list = get_clean_res_list(res_list, ensure_ca_exist=True)
            clean_res_full_id_list = [get_full_id_old(x.full_id, x.get_resname()) for x in clean_res_list] if (cfg_mode == "nciyes") else None

            if (cfg_mode == "tankbind"):
                _protein_dict = get_protein_feature(clean_res_list)
            elif (cfg_mode == "nciyes"):
                _protein_dict, _protein_res_id_dict = sx_get_protein_feature(clean_res_list, clean_res_full_id_list)
            elif (cfg_mode == "frag"):
                #TODO:write your function here
                try:
                    _protein_dict = get_protein_feature_qsar(clean_res_list)
                except Exception as e:
                    print(_pname+" ERROR_dim : "+str(e))
                
            ind = _pname[0]
            ind = int(ind) if str.isdigit(ind) else 10

            protein_dicts[ind][_pname] = _protein_dict
            if (cfg_mode == "nciyes"):
                protein_res_id_dict[_pname] = _protein_res_id_dict
                
        for _d in protein_dicts:
            protein_dict.update(_d)
        
        
        if cfg_save_files:
            os.system(f"mkdir -p {save_files_path}protein_dicts/")
            qrint(f"Saving {R}protein_dict{S} and {R}protein_res_id_dict{S}:" if (cfg_mode == "nciyes") else f"Saving {R}protein_dict{S}:")
            for i in tqdm(range(11), total=11):
                with open(f"{save_files_path}protein_dicts/dict_{str(i)}.pkl","wb") as f:
                    pickle.dump(protein_dicts[i], f)
            if (cfg_mode == "nciyes"):
                with open(f"{save_files_path}protein_dicts/res_dict.pkl","wb") as f:
                    pickle.dump(protein_res_id_dict, f)
        qrint(f"Successfully processed and saved {R}protein_dict{S} and {R}protein_res_id_dict{S} to {B}{save_files_path}protein_dicts/{S}." if (cfg_mode == "nciyes") 
            else f"Successfully processed and saved {R}protein_dict{S} to {B}{save_files_path}protein_dicts/{S}.")
        
    else:
        qrint(f"Loading {R}protein_dict{S} and {R}protein_res_id_dict{S} from {B}{save_files_path}protein_dicts/{S}" if (cfg_mode == "nciyes") 
            else f"Loading {R}protein_dict{S} from {B}{save_files_path}protein_dicts/{S}")
        for i in tqdm(range(11), total=11):
            with open(f"{save_files_path}protein_dicts/dict_{str(i)}.pkl","rb") as f:
                protein_dict.update(pickle.load(f))
        if (cfg_mode == "nciyes"):
            with open(f"{save_files_path}protein_dicts/res_dict.pkl","rb") as f:
                protein_res_id_dict.update(pickle.load(f))
        qrint(f"Successfully loaded {R}protein_dict{S} and {R}protein_res_id_dict{S}." if (cfg_mode == "nciyes")
            else f"Successfully loaded {R}protein_dict{S}.")
    protein_dicts = None



    # ### Segmentation of proteins by <font color=red>p2rank</font>


    # #### Generate or load <font color="red">.ds</font> file


    #time
    qrint("\n", r=False)
    qrint(f"{R}P2RANK{S}: {B}.ds{S} file generation")
    qrint(saveconfig(cfg_use_saved_files, cfg_save_files))

    if not cfg_use_saved_files:
        import shutil
        qrint(f"Processing {R}protein_list.ds{S}:")
        ds = f"{main_path}protein_list.ds"
        os.system(f"rm {ds}")
        with open(ds, "w") as out:
            for _fpath in tqdm(pdb_fpath_list, total=len(pdb_fpath_list)):
                out.write(f"{ds_path}{_fpath}\n")
        qrint(f"Successfully processed {R}protein_list.ds{S} at {B}{main_path}protein_list.ds{S}")
        if cfg_save_files:
            shutil.copy(ds, f"{save_files_path}protein_list.ds")
            qrint(f"Successfully saved {R}protein_list.ds{S} to {B}{save_files_path}protein_list.ds{S}")
    else:
        qrint(f"Using existing {R}protein_list.ds{S} file at {B}{save_files_path}protein_list.ds{S}.")
        ds = f"{save_files_path}protein_list.ds"


    # #### Generate or check <font color="red">p2rank results</font>


    #time
    qrint("\n", r=False)
    qrint(f"{R}P2RANK{S}: {B}p2rank{S} file generation or check")
    qrint(saveconfig(cfg_use_saved_files, cfg_save_files))
    _drop_p2rank = set()

    if not cfg_use_saved_files:
        if cfg_save_files:
            #os.mkdir(f"{save_files_path}p2rank")
            print(f"Running p2rank with output dir {B}{p2rank_save_path}{S}")
            cmd = f"{p2rank} predict {ds} -o {p2rank_save_path} -threads 8"
        else:
            os.system(f"mkdir -p {main_path}p2rank/")
            print(f"Running p2rank with output dir {B}{main_path}p2rank/{S}")
            cmd = f"{p2rank} predict {ds} -o {main_path}p2rank/ -threads 8"
        os.system(cmd)
        
    else:
        
        print(f"Checking existing p2rank files at {B}{p2rank_save_path}{S}:")
        _p2rank_dirset = set(os.listdir(f"{p2rank_save_path}"))
        _log = []
        for _pdb in pdb_code_list:
            if _pdb+"_protein.pdb_predictions.csv" not in _p2rank_dirset:
                _log.append(f"{_pdb}, prediction\n")
                _drop_p2rank.add(_pdb)
            if _pdb+"_protein.pdb_residues.csv" not in _p2rank_dirset:
                _log.append(f"{_pdb}, residues\n")
                _drop_p2rank.add(_pdb)
        if len(_log) == 0:
            qrint(f"Existing p2rank files at {B}{save_files_path}p2rank/{S} will be used.")
            
        else:
            with open(f"{main_path}log_absent_p2rank_pdbs.txt", "w") as f:
                f.writelines(_log)
            qrint(f"{len(_log)} files related to {len(_drop_p2rank)} proteins not found in {B}{save_files_path}p2rank/{S}.")
            qrint(f"Information of abasent files saved to {B}{main_path}log_absent_p2rank_pdbs.txt{S}.")
            qrint(f"Reprocessing {R}protein_list_absent.ds{S} for absent files:")
            ds2 = f"{main_path}log_absent_p2rank_pdbs.txt"
            with open(ds, "r") as infile:
                with open(ds2, "w") as out:
                    for _line in infile.readlines():
                        if re.match(r".+\/(....)_protein.pdb", _line).groups()[0] in _drop_p2rank:
                            out.write(_line)
            qrint(f"Successfully processed {R}protein_list_absent.ds{S} at {B}{main_path}protein_list_absent.ds{S}")
            
            qrint(f"Running p2rank with output dir {B}{main_path}p2rank/{S}")
            cmd = f"{p2rank} predict {ds2} -o {main_path}p2rank/ -threads 8"
            os.system(f"mkdir -p {main_path}p2rank")
            os.system(cmd)
            import shutil
            qrint(f"Copying generated files to {B}{save_files_path}p2rank/{S}")
            for _f in os.listdir(f"{main_path}p2rank/"):
                if _f != "params.txt" and _f != "visualizations" and _f != "run.log":
                    shutil.copy(f"{main_path}p2rank/{_f}", f"{p2rank_save_path}{_f}")
            for _f in os.listdir(f"{main_path}p2rank/visualizations/"):
                if _f != "data":
                    shutil.copy(f"{main_path}p2rank/visualizations/{_f}", f"{p2rank_save_path}visualizations/{_f}")
            for _f in os.listdir(f"{main_path}p2rank/visualizations/data"):
                    shutil.copy(f"{main_path}p2rank/visualizations/data/{_f}", f"{p2rank_save_path}visualizations/data/{_f}")


    # ### Get ligand infomation: <font color="red">ligand_df</font> (& <font color="red">ligand_atom_id_dict</font>)


    #time
    qrint("\n", r=False)
    qrint(f"{R}ligand_df{S} & {R}ligand_atom_id_dict{S} processing" if (cfg_mode=="nciyes") else f"{R}ligand_df{S} processing")
    qrint(saveconfig(cfg_use_saved_files, cfg_save_files))


    ligand_atom_id_dict = {} if (cfg_mode=="nciyes") else None
        
    parser = PDBParser()
    if not cfg_use_saved_files:
        print(f"Processing {R}ligand_df{S}:")
        if cfg_mode == "nciyes":
            ligand_name_list = list(ligand_df["ligand_name_ncifile"])
        ligand_pdb_list = list(ligand_df["pdb_code"])
        ligand_fpath_list = list(ligand_df["ligand_fpath"])
        canonique_smiles = []
        error_list = []
        for _fpath, _pdb in tqdm(zip(ligand_fpath_list, ligand_pdb_list), total=len(ligand_fpath_list)):
            if ".sdf" in _fpath:
                try:
                    canonique_smiles.append(get_canonical_smiles(Chem.MolToSmiles(Chem.MolFromMolFile(_fpath))))
                except Exception as e:
                    canonique_smiles.append(f"ERROR - {str(e)}")
                    error_list.append(_fpath)
                    continue            
            elif ".pdb" in _fpath:
                try:
                    canonique_smiles.append(get_canonical_smiles(Chem.MolToSmiles(Chem.MolFromPDBFile(_fpath))))
                except Exception as e:
                    canonique_smiles.append(f"ERROR - {str(e)}")
                    error_list.append(_fpath)
                    continue
            if cfg_mode == "nciyes":
                structure = parser.get_structure("pdb", _fpath)[0]
                _atom_ids = []
                for chain in structure:
                    for residue in chain:
                        for atom in residue.get_atoms():
                            atom_id = atom.get_name()
                            _atom_ids.append(atom_id)
                ligand_atom_id_dict[_pdb] = {_atom:i for (_atom, i) in zip(_atom_ids, range(len(_atom_ids)))}
        ligand_df["canonique_smiles"] = canonique_smiles
        qrint(f"Successfully processed {R}ligand_df{S}.")
        
        if cfg_save_files:
            # Saving ligand_df
            ligand_df.to_csv(f"{save_files_path}ligand_df.csv")
            qrint(f"Successfully saved {R}ligand_df{S} to {B}{save_files_path}ligand_df.csv{S}.")
            if cfg_mode == "nciyes":
                with open(f"{save_files_path}ligand_atom_id_dict.pkl", "wb") as f:
                    pickle.dump(ligand_atom_id_dict, f)
                qrint(f"Successfully saved {R}ligand_atom_id_dict{S} to {B}{save_files_path}ligand_atom_id_dict.pkl{S}.")

    else: # cfg_use_saved_files = True
        qrint(f"Loading {R}ligand_df{S} from {B}{save_files_path}ligand_df.csv{S}:")
        ligand_df = pd.read_csv(f"{save_files_path}ligand_df.csv")
        qrint(f"Successfully loaded {R}ligand_df{S}.")
        
        if cfg_mode == "nciyes":
            qrint(f"Loading {R}ligand_atom_id_dict{S} from {B}{save_files_path}ligand_atom_id_dict.pkl{S}:")
            with open(f"{save_files_path}ligand_atom_id_dict.pkl", "rb") as f:
                ligand_atom_id_dict = pickle.load(f)
            qrint(f"Successfully loaded {R}ligand_atom_id_dict{S}.")


    ligand_df_without_smiles = ligand_df[ligand_df.canonique_smiles.str.contains("ERROR")]
    ligand_df_without_smiles.to_csv(f"{save_files_path}MDropped_1.Ligands_{len(ligand_df_without_smiles)} - SMILES Generation Error.csv")
    ligand_df_with_smiles = ligand_df[~ligand_df.canonique_smiles.str.contains("ERROR")]
    ligand_df_with_smiles.to_csv(f"{save_files_path}ligand_df_with_SMILES.csv")


    # ### Get <font color = "red"> info </font> for dataset processing


    #time
    qrint("\n", r=False)
    qrint(f"{R}info{S} for dataset processing")
    qrint(saveconfig(cfg_use_saved_files, cfg_save_files))

    if not cfg_use_saved_files:
        qrint(f"Processing {R}info{S}:")
        info = []
        error_info = []
        for i,line in tqdm(ligand_df_with_smiles.iterrows(), total=ligand_df_with_smiles.shape[0]):
            ligand_name = line["ligand_name_file"]
            pdb_code = line["pdb_code"]
            pdb_fname = line["pdb_code"] + "_protein"
            canonique_smiles = line["canonique_smiles"]
            ligand_fpath = line["ligand_fpath"]
            ligand_ftype = os.path.splitext(os.path.split(ligand_fpath)[1])[1]
            release_year = line["release_year"]
            affinity = line["affinity"]
            if cfg_save_files or cfg_use_saved_files:
                p2rankFile = f"{p2rank_save_path}{pdb_fname}.pdb_predictions.csv"
            else:
                p2rankFile = f"{main_path}p2rank/{pdb_fname}.pdb_predictions.csv"
                        
            pocket_df = pd.read_csv(p2rankFile)
            pocket_df.columns = pocket_df.columns.str.strip()
            pocket_coms = pocket_df[["center_x", "center_y", "center_z"]].values
            if len(pocket_coms) == 0:
                error_info.append([pdb_code, "p2rank file error", ligand_name, ligand_fpath, release_year])
                continue
            if cfg_right_pocket_by_minor_distance:
                if ligand_ftype == ".sdf":
                    coord = Chem.MolFromMolFile(ligand_fpath).GetConformer().GetPositions()
                elif ligand_ftype == ".pdb":
                    coord = Chem.MolFromPDBFile(ligand_fpath).GetConformer().GetPositions()
                coord = coord.sum(axis=0)/len(coord)
                right_ith_pocket = ((pocket_coms-coord)**2).sum(axis=1).argmin()
            for ith_pocket, _pocket_com in enumerate(pocket_coms):
                _pocket_com = ",".join([str(a.round(3)) for a in _pocket_com])
                if cfg_right_pocket_by_minor_distance:
                    right_pocket_by_distance = (ith_pocket == right_ith_pocket)
                    info.append([ligand_name, pdb_code, canonique_smiles, ligand_fpath, ligand_ftype, affinity, f"pocket_{ith_pocket+1}", _pocket_com, release_year, right_pocket_by_distance])
                else:
                    info.append([ligand_name, pdb_code, canonique_smiles, ligand_fpath, ligand_ftype, f"pocket_{ith_pocket+1}", _pocket_com, release_year])
                '''
                if False:
                    #except Exception as e:
                    _pdb = line["pdb_code"]
                    ligand_name = line["ligand_name_ncifile"]
                    pdb_code = line["pdb_code"]
                    pdb_fname = line["pdb_code"] + "_protein"
                    canonique_smiles = line["canonique_smiles"]
                    ligand_fpath = line["ligand_fpath"]
                    ligand_ftype = os.path.splitext(os.path.split(ligand_fpath)[1])[1]
                    pocket_df = pd.read_csv(p2rankFile)
                    pocket_df.columns = pocket_df.columns.str.strip()
                    pocket_coms = pocket_df[["center_x", "center_y", "center_z"]].values
                    affinity = line["affinity"]
                    release_year = line["release_year"]
                    #print(f"Error with {_pdb}: {str(e)} : {ligand_name},{pdb_fname},{canonique_smiles},{ligand_fpath},{ligand_ftype},{pocket_coms},{affinity}")
                    error_info.append([_pdb, str(e), ligand_name, pdb_fname, canonique_smiles, ligand_fpath, ligand_ftype, pocket_coms, affinity, release_year])
                '''
        info = pd.DataFrame(info, columns = ["ligand_name", "pdb_code", "canonique_smiles", 
                                            "ligand_fpath", "ligand_ftype", "affinity", "pocket_name", "pocket_com", "release_year", "right_pocket_by_distance"])
        error_info = pd.DataFrame(error_info, columns = ["pdb_code", "info", "ligand_name", "ligand_fpath", "release_year"])
        qrint(f"Successfully processed {R}info{S}.")
        
        # Remove recorded bad info (可能没用，可以不管）
        if os.path.exists(f"{save_files_path}MDropped_InModel - Dropped_pocket.csv"):
            _drop_pocket = pd.read_csv(f"{save_files_path}MDropped_InModel - Dropped_pocket.csv")
            for i in range(len(_drop_pocket)):
                line = _drop_pocket.iloc[i]
                _d_pdb, _d_ligname, _d_pocket = line["pdb_code"], line["ligand_name"], line["pocket"]
                info = info.drop(info[(info.pdb_code == _d_pdb) & (info.ligand_name == _d_ligname) & (info.pocket_name == _d_pocket)].index)
        if os.path.exists(f"{save_files_path}MDropped_InModel - Dropped_protein.txt"):
            with open(f"{save_files_path}MDropped_InModel - Dropped_protein.txt", "r") as f:
                _drop_proteins = f.readlines()
            _drop_proteins = [_dp.replace("\n", "") for _dp in _drop_proteins]
            info = info[~info["pdb_code"].isin(_drop_proteins)]
        info.reset_index()
        if "index" in info.columns:
            del info["index"]
        if cfg_save_files:
            info.to_csv(f"{save_files_path}info.csv")
            qrint(f"Successfully saved {R}info{S} to {B}{save_files_path}info.csv{S}.")
            error_info.to_csv(f"{save_files_path}MDropped_2.PDBs - Info Generation Error.csv")
            qrint(f"Successfully saved {R}error_info{S} to {B}{save_files_path}MDropped_2.PDBs - Info Generation Error.csv{S}.")

    else:
        qrint(f"Loading {R}info{S} from {B}{save_files_path}info.csv{S}:")
        info = pd.read_csv(f"{save_files_path}info.csv", index_col = 0)
        
        if os.path.exists(f"{save_files_path}MDropped_InModel - Dropped_pocket.csv"):
            _drop_pocket = pd.read_csv(f"{save_files_path}MDropped_InModel - Dropped_pocket.csv")
            for i in range(len(_drop_pocket)):
                line = _drop_pocket.iloc[i]
                _d_pdb, _d_ligname, _d_pocket = line["pdb_code"], line["ligand_name"], line["pocket"]
                info = info.drop(info[(info.pdb_code == _d_pdb) & (info.ligand_name == _d_ligname) & (info.pocket_name == _d_pocket)].index)
            qrint(f"Removed bad input from {B}{save_files_path}MDropped_InModel - Dropped_pocket.csv{S}.")
        if os.path.exists(f"{save_files_path}MDropped_InModel - Dropped_protein.txt"):
            with open(f"{save_files_path}MDropped_InModel - Dropped_protein.txt", "r") as f:
                _drop_proteins = f.readlines()
            _drop_proteins = [_dp.replace("\n", "") for _dp in _drop_proteins]
            info = info[~info["pdb_code"].isin(_drop_proteins)]
            qrint(f"Removed bad input from {B}{save_files_path}MDropped_InModel - Dropped_protein.txt{S}.")
        if os.path.exists(f"{save_files_path}MDropped_InModel - Incoherent Res Name in old NCI files.csv"):    
            _drop_pocket = pd.read_csv(f"{save_files_path}MDropped_InModel - Incoherent Res Name in old NCI files.csv")
            for i in range(len(_drop_pocket)):
                line = _drop_pocket.iloc[i]
                _d_pdb, _d_ligname, _d_pocket = line["pdb_code"], line["ligand_name"], line["pocket"]
                info = info.drop(info[(info.pdb_code == _d_pdb) & (info.ligand_name == _d_ligname) & (info.pocket_name == _d_pocket)].index)
            qrint(f"Removed bad input from {B}{save_files_path}MDropped_InModel - Incoherent Res Name in old NCI files.csv{S}.")
        
        info = info.reset_index()
        if "index" in info.columns:
            del info["index"]
        
        qrint(f"Successfully loaded {R}info{S}. Length: {B}{len(info)}{S}")


    # ### Construct dataset
    import torch
    torch.set_num_threads(1)
    from torch_geometric.data import Dataset
    import rdkit.Chem as Chem    # conda install rdkit -c rdkit if import failure.
    from feature_utils import extract_torchdrug_feature_from_mol

    if cfg_mode == "tankbind":
        from utils import construct_data_from_graph_gvp
    if cfg_mode == "nciyes":
        from sx_utils import sx_construct_data_from_graph_gvp
        from sx_feature_utils import sx_extract_torchdrug_feature_from_mol
        from sx_new_utils import sx_ligand_dedocking
        from sx_new_utils import sx_get_nci_matrix_by_dict
    elif cfg_mode == "frag":
        from utils import construct_data_from_graph_gvp
        nci_df = None
        #TODO: import your functions here.

    #TODO: move all these import to the beginning of this notebook when all codes are finished.

    class MyDataset_VS(Dataset):
        def __init__(self, root, data=None, protein_dict=None, proteinMode=0, compoundMode=1,
                    pocket_radius=20, shake_nodes=None, 
                    transform=None, pre_transform=None, pre_filter=None, generate_3D_conf = False,
                    protein_res_id_dict=None, nci_df=None, ligand_atom_id_dict=None, cfg_mode=None,
                    right_pocket_by_distance=True,
                    ):
            self.data = data
            self.protein_dict = protein_dict
            super().__init__(root, transform, pre_transform, pre_filter)
            print(self.processed_paths)
            self.data = torch.load(self.processed_paths[0])
            self.protein_dict = torch.load(self.processed_paths[1])
            self.nci_df=nci_df
            self.protein_res_id_dict = protein_res_id_dict
            self.ligand_atom_id_dict = ligand_atom_id_dict
            self.proteinMode = proteinMode
            self.pocket_radius = pocket_radius
            self.compoundMode = compoundMode
            self.shake_nodes = shake_nodes
            self.generate_3D_conf = generate_3D_conf
            self.cfg_mode = cfg_mode if cfg_mode else "tankbind"
            self.right_pocket_by_distance=right_pocket_by_distance
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
            canonique_smiles = line['canonique_smiles']
            pocket_com = line['pocket_com']
            pocket_com = np.array(pocket_com.split(",")).astype(float) if type(pocket_com) == str else pocket_com
            pocket_com = pocket_com.reshape((1, 3))
            use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False
            protein_name = line['pdb_code']
            pocket_name = line['pocket_name']
            protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[protein_name]
            ligand_fpath = line['ligand_fpath']
            ligand_ftype = line['ligand_ftype']
            ligand_name = line['ligand_name']
            affinity = line['affinity']
            
            if self.right_pocket_by_distance:
                right_pocket_by_distance = line['right_pocket_by_distance']
            
            if ligand_ftype == ".pdb":
                mol = Chem.MolFromPDBFile(ligand_fpath)
            elif ligand_ftype == ".sdf":
                mol = Chem.MolFromMolFile(ligand_fpath)
            mol.Compute2DCoords()  
            
            if self.cfg_mode == "tankbind":
                try:
                    coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
                except Exception as e:
                    return protein_name+" ERROR_II : "+str(e)
                try:
                    data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                        protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                                        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                                        pocket_radius=self.pocket_radius, use_whole_protein=use_whole_protein, includeDisMap=True,
                                        use_compound_com_as_pocket=False, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)
                    data.compound_pair = pair_dis_distribution.reshape(-1, 16)
                except Exception as e:
                    return protein_name+" ERROR_III : "+str(e)
                try:
                    data.right_pocket_by_distance = right_pocket_by_distance
                    data.affinity = affinity
                    data.dataname = protein_name + "_" + ligand_name + "_" + pocket_name
                except Exception as e:
                    return protein_name+" ERROR_IV : "+str(e)
            
            elif self.cfg_mode == "nciyes":
                protein_res_ids = self.protein_res_id_dict[protein_name]
                if (len(protein_res_ids.keys())-1) != protein_res_ids[list(protein_res_ids.keys())[-1]]:
                    return protein_name+" ERROR_I : protein_res_ids length error."
                try:
                    coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = sx_extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True, generate_3D_conf=False)
                except Exception as e:
                    return protein_name+" ERROR_II : "+str(e)
                    # y is distance map, instead of contact map.
                try:
                    data, input_node_list, keepNode = sx_construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                        protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                                        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                                        pocket_radius=self.pocket_radius, use_whole_protein=use_whole_protein, includeDisMap=True,
                                        use_compound_com_as_pocket=False, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)
                except Exception as e:
                    return protein_name+" ERROR_III : "+str(e)
                data.compound_pair = pair_dis_distribution.reshape(-1, 16)
                kept_res_ids = [_id for (_id, _keep) in zip(protein_res_ids, keepNode) if _keep]

                try:
                    atom_ids = self.ligand_atom_id_dict[protein_name]
                    #data.nci_sequence = torch.Tensor(sx_get_nci_matrix_by_dict(protein_name, ligand_name, res_full_id, atom_ids, self.nci_df).flatten())
                    data.nci_sequence = torch.tensor(sx_get_nci_matrix_by_dict(protein_name, ligand_name, kept_res_ids, atom_ids, self.nci_df).flatten())
                    data.pair_shape = (len(kept_res_ids), len(atom_ids))
                    data.right_pocket_by_distance = right_pocket_by_distance
                    data.affinity = affinity
                    data.dataname = protein_name + "_" + ligand_name + "_" + pocket_name
                except Exception as e:
                    return protein_name+" ERROR_IV : "+str(e)
            
            elif self.cfg_mode == "frag":
                try:
                    coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
                except Exception as e:
                    return protein_name+" ERROR_II : "+str(e)
                try:
                    data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                        protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                                        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                                        pocket_radius=self.pocket_radius, use_whole_protein=use_whole_protein, includeDisMap=True,
                                        use_compound_com_as_pocket=False, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)
                    data.compound_pair = pair_dis_distribution.reshape(-1, 16)
                except Exception as e:
                    return protein_name+" ERROR_III : "+str(e)
                try:
                    data.right_pocket_by_distance = right_pocket_by_distance
                    data.affinity = affinity
                    data.dataname = protein_name + "_" + ligand_name + "_" + pocket_name
                except Exception as e:
                    return protein_name+" ERROR_IV : "+str(e)
                #TODO: write your codes here. Refer to "if self.cfg_mode=="tankband" above.

            return data


    # #### data split


    # ### datasets generation or load


    #time
    qrint("\n", r=False)
    qrint(f"{R}dataset{S} generation or load")
    qrint(saveconfig(cfg_use_saved_files, cfg_save_files))
    qrint("\n", r=False)
    qrint(f"{R}info split{S} for dataset processing")


    if cfg_timesplit: # cfg_timesplit = True
        qrint(f"{R}Timesplit{S}: for dataset processing, the results will always be saved.")
        qrint(f"Checking timesplit files：")
        if not os.path.exists(f"{save_files_path}timesplit_train_no_lig_overlap.txt") or not os.path.exists(f"{save_files_path}timesplit_val_no_lig_overlap.txt") or not os.path.exists(f"{save_files_path}timesplit_test.txt"):
            qrint(f"Timesplit files not found. {R}Length split strategy{S} will be applied.")
            cfg_timesplit = False
        else:
            qrint(f"Found timesplit files. {R}Timesplit{S} will be applied.")
            with open(f"{save_files_path}timesplit_train_no_lig_overlap.txt", "r") as f:
                _train = [_t.replace("\n", "") for _t in f.readlines()]
            with open(f"{save_files_path}timesplit_val_no_lig_overlap.txt", "r") as f:
                _val = [_t.replace("\n", "") for _t in f.readlines()]
            with open(f"{save_files_path}timesplit_test.txt", "r") as f:
                _test = [_t.replace("\n", "") for _t in f.readlines()]
                
            dataset_path = f"{save_files_path}dataset/"
            qrint(f"Processing {R}dataset{S}:")
            if not cfg_use_saved_files:
                os.system(f"rm -r {dataset_path}")
                for _s in ["train", "val", "test", "test2"]:
                    os.system(f"mkdir -p {dataset_path}{_s}")
                info_test2 = info[~(info.pdb_code.isin(_train)|info.pdb_code.isin(_val)|info.pdb_code.isin(_test))]
                test2_set = MyDataset_VS(f"{dataset_path}test2/", data=info_test2, protein_dict=protein_dict, 
                                        protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode) if len(info_test2) \
                            else None        
            else:
                test2_set = MyDataset_VS(f"{dataset_path}test2/", data=info[~(info.pdb_code.isin(_train)|info.pdb_code.isin(_val)|info.pdb_code.isin(_test))], 
                                        protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode) \
                            if os.path.exists(f"{dataset_path}test2/processed/data.pt") else None
            
            train_set = MyDataset_VS(f"{dataset_path}train/", data=info[info.pdb_code.isin(_train)], protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode)
            val_set = MyDataset_VS(f"{dataset_path}val/", data=info[info.pdb_code.isin(_val)], protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode)
            test_set = MyDataset_VS(f"{dataset_path}test/", data=info[info.pdb_code.isin(_test)], protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode)
                
    if not cfg_timesplit:
        test2_set = None
        train_part, val_part, test_part = cfg_split_strategy[0], cfg_split_strategy[1], cfg_split_strategy[2]
        train_val_split, val_test_split = int(train_part * len(info)), int((train_part+val_part) * len(info))
        qrint(f"{R}Length% split strategy{S}: {train_val_split} - {val_test_split-train_val_split} - {len(info)-val_test_split}. For dataset processing, the results will always be saved.")
        dataset_path = f"{save_files_path}dataset/"
        qrint(f"Processing {R}dataset{S}:")
        if not cfg_use_saved_files:
            os.system(f"rm -r {dataset_path}")
            for _s in ["train", "val", "test", "test2"]:
                os.system(f"mkdir -p {dataset_path}{_s}")
        train_set = MyDataset_VS(f"{dataset_path}train/", data=info.iloc[0:train_val_split], protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode)
        val_set = MyDataset_VS(f"{dataset_path}val/", data=info.iloc[train_val_split:val_test_split], protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode)
        test_set = MyDataset_VS(f"{dataset_path}test/", data=info.iloc[val_test_split:], protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode)
        
    qrint(f"Successfully processed {R}dataset{S}s.")
    qrint(f"-- {R}train set{S} : {len(train_set)}")
    qrint(f"-- {R}val set{S}   : {len(val_set)}")
    qrint(f"-- {R}test set{S}  : {len(test_set)}")
    qrint(f"-- {R}test2 set{S} : {len(test2_set) if test2_set else 0}")


    def checkdata2(data, fpath = "./trash.txt", cfg_mode=cfg_mode):
        if isinstance(data, str) or isinstance(data, list):
            with open(fpath, "a") as f:
                f.write(f"{data} error!\n")        
        elif cfg_mode=="nciyes":
            device = data.y_batch.device
            samples = []
            shapes = []
            ff = True

            # check nci record in right_pocket
            for i, _right in enumerate(data.right_pocket_by_distance):
                if _right == True:
                    samples.append(i)
                samples = torch.tensor(samples).to(device)
            index = torch.isin(data.y_batch, samples)
            ss = data.nci_sequence[index]
            if len(samples) and not (ss.sum()):
                with open(fpath, "a") as f:
                    f.write(f"{data.dataname} : its right pocket has no NCI record!\n")


    # ## Training Block


    import logging
    from torch_geometric.loader import DataLoader

    batch_size = args.batch_size
    learning_rate = args.lr
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    if cfg_mode == "tankbind":
        from model import get_model
        model = get_model(0, logging, device)
        modelFile = "../saved_models/self_dock.pt"
        model.load_state_dict(torch.load(modelFile, map_location=device))
        
    if cfg_mode == "nciyes":
        from sx_model import sx_get_model
        model = sx_get_model(0, logging, device, nciyes=True, margin=1, margin_weight=1, nci_weight=1, output_classes=2,
                            class_weight=torch.tensor([1,1],dtype=torch.float32))
        IaBNetFile = "../saved_models/self_dock.pt"
        model.IaBNet.load_state_dict(torch.load(IaBNetFile, map_location=device))
        
    elif cfg_mode == "frag":
        #TODO: WRITE YOUR CODE HERE
        #from your_model_file import your_model
        from model_frag import get_model
        model = get_model(0, logging, device)
        # modelFile = "../saved_models/self_dock.pt"
        # model.load_state_dict(torch.load(modelFile, map_location=device))

    logging.basicConfig(level=logging.INFO)
    _ = model.eval()

    train_data_loader = DataLoader(train_set, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=True, num_workers=args.num_workers)
    val_data_loader = DataLoader(val_set, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=args.num_workers)  if len(val_set) else None
    test_data_loader = DataLoader(test_set, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=args.num_workers) if len(test_set) else None
    test2_data_loader = DataLoader(test2_set, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=args.num_workers) if test2_set else None
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    list_val_metric = []


    for _epoch in range(args.max_epoch):
        qrint(f"Training Epoch: {_epoch}")
        model.train()
        _list_loss = []
        _list_affinity = []
        errors, error_file = [], f"{main_path}train_{_epoch}_errors.txt"
        for i,data in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            # qrint(f"Train batch {i}", r=False, newline=False)
            if cfg_checkdata2:
                checkdata2(data)
                if isinstance(data, str) or isinstance(data, list):
                    errors.append(data)
                    with open(error_file, "a") as f:
                        f.write("\n"+str(data)+"\n")    
                    continue
            data = data.to(device)
            if cfg_mode == "tankbind": # Results : affinity_pred_dictbb
                y_pred, affinity_pred = model(data)
            elif cfg_mode == "frag": # Results : many
                data = data.to(device)
                y_pred, affinity_pred = model(data)
                loss = model.calculate_loss(affinity_pred, y_pred,  data.affinity, data.dis_map,
                                            data.right_pocket_by_distance)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _list_loss.append(loss.detach().cpu())
                _list_affinity.append(affinity_pred.detach().cpu())
        
        with open (f"{save_files_path}affinity_dict_train_{_epoch}.pkl", "wb") as f:
            pickle.dump(_list_affinity, f)
        
        losst = torch.cat([torch.tensor([i]) for i in _list_loss])
        losst = torch.mean(losst, dim=0)
        qrint("\n")
        qrint(f"Loss: {losst.item()}")
        print('TRAIN----> Epoch [{}/{}], Loss: {:.4f}' .format(_epoch+1, args.max_epoch, losst.item()))
        model.eval()
        
        if val_data_loader is not None:
            with torch.no_grad():
                qrint(f"Val batch {i}", r=False, newline=False)
                _list_loss_va = []
                _list_affinity = []
                for data in tqdm(val_data_loader):
                    data = data.to(device)
                    if cfg_mode == "tankbind": # Results : affinity_pred_dict
                        y_pred, affinity_pred = model(data)
                    elif cfg_mode == "frag": # Results : many
                        y_pred, affinity_pred = model(data)
                        loss = model.calculate_loss(affinity_pred, y_pred,  data.affinity, data.dis_map,
                                                    data.right_pocket_by_distance)
                        _list_loss_va.append(loss.detach().cpu())
                        _list_affinity.append(affinity_pred.detach().cpu())
                losst_va = torch.cat([torch.tensor([i]) for i in _list_loss_va])
                losst_va = torch.mean(losst_va, dim=0)
                with open (f"{save_files_path}affinity_dict_val_{_epoch}.pkl", "wb") as f:
                    pickle.dump(_list_affinity, f)
                print('VALID----> Epoch [{}/{}], Loss: {:.4f}' .format(_epoch+1, args.max_epoch, losst_va.item()))
                qrint("\n")
                qrint(f"Loss: {losst_va.item()}")
                state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': _epoch}
                save_model_path = save_model_path + '/%s/lr%s-batchsize%s-%s' % (args.iter, args.lr, args.batch_size, args.config_mode)
                if not os.path.exists(save_model_path):
                    os.system(f"mkdir -p {save_model_path}")
                epoch_save_model_path = '%s/epoch-%s' % (save_model_path, _epoch + 1)
                torch.save(state, epoch_save_model_path)
    model.eval() 
    with torch.no_grad():
        ## TODO 定义一个score用来评估
        if test_data_loader is not None:
            _list_loss_te = []
            _list_affinity = []
            for data in tqdm(test_data_loader):
                data = data.to(device)
                if cfg_mode == "tankbind": # Results : affinity_pred_dict
                        data = data.to(device)
                        y_pred, affinity_pred = model(data)
                elif cfg_mode == "frag": # Results : many
                    y_pred, affinity_pred = model(data)
                    loss = model.calculate_loss(affinity_pred, y_pred,  data.affinity, data.dis_map,
                            data.right_pocket_by_distance)
                    _list_loss_te.append(loss.detach().cpu())
                    _list_affinity.append(affinity_pred.detach().cpu())
            with open (f"{save_files_path}affinity_dict_test_{_epoch}.pkl", "wb") as f:
                pickle.dump(_list_affinity, f)
                
            losst_te = torch.cat([torch.tensor([i]) for i in _list_loss_te])
            losst_te = torch.mean(losst_te, dim=0)
            qrint('TEST----> loss: {}' .format(losst_te))
        
        if test2_data_loader is not None:
            _list_loss_te_2 = []
            for data in tqdm(test2_data_loader):
                if cfg_mode == "tankbind": # Results : affinity_pred_dict
                        data = data.to(device)
                        y_pred, affinity_pred = model(data)
                elif cfg_mode == "frag": # Results : many
                    y_pred, affinity_pred = model(data)
                    loss = model.calculate_loss(affinity_pred, y_pred,  data.affinity, data.dis_map,
                            data.right_pocket_by_distance)
                    _list_loss_te_2.append(loss.detach().cpu())
            losst_te_2 = torch.cat([torch.tensor([i]) for i in _list_loss_te_2])
            losst_te_2 = torch.mean(losst_te_2, dim=0)
            qrint('TEST2----> loss: {}' .format(losst_te_2))

        

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default='data')

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout_rate", type=float, default=0)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--config_mode", choices=['tankbind', "nciyes", "frag"], default='frag')
    parser.add_argument("--data_version", type=str, default='v1.0')    
    parser.add_argument("--first_run", type=bool, default=False) 
    args = parser.parse_args()
    
    main(args)







