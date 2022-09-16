
def main(args):
    
    import time
    time.strftime("%H:%M:%S")

    # %% [markdown]
    # ## Configuration and Initilization
    # ### Configuration

    # %% [markdown]
    # 1st run:
    # - set "cfg_save_files = True" and "cfg_use_saved_files = False" to process data and saved them.
    # 
    # otherwise:
    # - set "cfg_save_files = False" and "cfg_use_saved_files = True" to skip processing and load saved data.

    # %%
    # You may want to change settings below.
    cfg_use_saved_files = True # If you have already generated all the data required.
    cfg_save_files = False # If you want to save generated data.
    cfg_checkdata = True
    cfg_split_strategy = (0.5,0.25,0.25)

    cfg_data_version = "v9.8-only2" # "v9.8-only2"
    cfg_timesplit = True # If timesplit is applicated for splitting the dataset.
    cfg_custom_dir_name = "SJ-NCIYES" # Prefix in the name of the output folder.
    cfg_distinguish_by_timestamp = True # If true, a timestamp is added to your output dir name.


    # Choose one
    cfg_mode = args.config_mode
    cfg_data_version = args.data_version

    # Chose one
    cfg_running_mode = ["train", "inference"]
    cfg_running_mode = "train"
    #save_model_path + '/%s/lr%s-batchsize%s-%s' % (args.iter, args.lr, args.batch_size, args.model_mode)

    # %% [markdown]
    # ### Other Configuration and Initilization
    # Under normal condition, you needn't modify this chapiter.

    # %%
    # Under normal condition, you should not modify the codes below.
    input_path = f"../../TankBind/ours/Inputs/{cfg_mode}/{cfg_data_version}/"
    output_path = f"./Outputs/{cfg_mode}/"
    save_files_path = f"../../TankBind/ours/Inputs/Savedfiles/{cfg_mode}.{cfg_data_version}/"
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
    cfg_jupyter = args.jupyter # If you're using jupyter notebook. # old
    cfg_right_pocket_by_minor_distance = True # If the right pocket is chosen by calculating distance between ligand center and pocket center.

    # %%
    import os
    import time
    import pickle
    import sys
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from Bio.PDB import PDBParser
    import re
    sys.path.insert(0, "../tankbind/")

    if cfg_mode == "tankbind":
        pass
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

    def qrint(target, jupyter= cfg_jupyter, log=log_fpath, R=R, B=B, S=S, r=True):
        if r:
            print(target)
        if jupyter:
            target = target.replace(R,"").replace(B,"").replace(S,"")
        with open(log_fpath, "a") as f:
            if target != "\n":
                f.write(time.strftime("[%m-%d %H:%M:%S] ")+target.replace("\n", "                 \n")+"\n")
            else:
                f.write("\n")

    if cfg_mode == "nciyes":
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

    if os.path.exists(datainfo):
        with open(datainfo,"r") as f:
            _ls = [k.replace("\n","") for k in f.readlines()]
            for _l in _ls:
                qrint(_l)
            qrint("\n")
        
    pdb_df = pd.read_csv(pdb_df_fpath, index_col=0)
    ligand_df = pd.read_csv(ligand_df_fpath, index_col=0)
    pdb_code_list = list(pdb_df["pdb_code"])
    pdb_fpath_list = list(pdb_df["pdb_fpath"])
    qrint(f"Loaded {R}pdb_df{S} from {B}{pdb_df_fpath}{S} with length {len(pdb_df)}")
    qrint(f"Loaded {R}ligand_df{S} from {B}{ligand_df_fpath}{S} with length {len(ligand_df)}")
    if cfg_mode == "nciyes":
        nci_df = pd.read_csv(nci_df_fpath, index_col=0)
        qrint(f"Loaded {R}nci_df{S} from {B}{nci_df_fpath}{S} with length {len(nci_df)}")

    # %%
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"{main_path}")

    # %% [markdown]
    # ## Running!

    # %% [markdown]
    # ### Get protein features: <font color="red">protein_dict</font> (& <font color="red">protein_res_id_dict</font>)

    # %%

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
        with open(f"{save_files_path}protein_dicts/res_dict.pkl","rb") as f:
            protein_res_id_dict.update(pickle.load(f))
        qrint(f"Successfully loaded {R}protein_dict{S} and {R}protein_res_id_dict{S}." if (cfg_mode == "nciyes")
            else f"Successfully loaded {R}protein_dict{S}.")

    # %% [markdown]
    # ### Segmentation of proteins by <font color=red>p2rank</font>

    # %% [markdown]
    # #### Generate or load <font color="red">.ds</font> file

    # %%

    cfg_use_saved_files = True
    cfg_save_files = False
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

    # %% [markdown]
    # #### Generate or check <font color="red">p2rank results</font>

    # %%

    cfg_use_saved_files = True
    cfg_save_files = False
    qrint("\n", r=False)
    qrint(f"{R}P2RANK{S}: {B}p2rank{S} file generation or check")
    qrint(saveconfig(cfg_use_saved_files, cfg_save_files))
    _drop_p2rank = set()

    if not cfg_use_saved_files:
        if cfg_save_files:
            #os.mkdir(f"{save_files_path}p2rank")
            qrint(f"Running p2rank with output dir {B}{p2rank_save_path}{S}")
            cmd = f"{p2rank} predict {ds} -o {p2rank_save_path} -threads 8"
        else:
            os.system(f"mkdir -p {main_path}p2rank/")
            qrint(f"Running p2rank with output dir {B}{main_path}p2rank/{S}")
            cmd = f"{p2rank} predict {ds} -o {main_path}p2rank/ -threads 8"
        os.system(cmd)
        
    else:
        
        qrint(f"Checking existing p2rank files at {B}{p2rank_save_path}{S}:")
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

    # %% [markdown]
    # ### Get ligand infomation: <font color="red">ligand_df</font> (& <font color="red">ligand_atom_id_dict</font>)

    # %%

    qrint("\n", r=False)
    qrint(f"{R}ligand_df{S} & {R}ligand_atom_id_dict{S} processing" if (cfg_mode=="nciyes") else f"{R}ligand_df{S} processing")
    qrint(saveconfig(cfg_use_saved_files, cfg_save_files))


    ligand_atom_id_dict = {} if (cfg_mode=="nciyes") else None
        
    parser = PDBParser()
    if not cfg_use_saved_files:
        qrint(f"Processing {R}ligand_df{S}:")
        if cfg_mode == "nciyes":
            ligand_name_list = list(ligand_df["ligand_name_file"])
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

    # %%
    ligand_df_without_smiles = ligand_df[ligand_df.canonique_smiles.str.contains("ERROR")]
    ligand_df_without_smiles.to_csv(f"{save_files_path}MDropped_1.Ligands_{len(ligand_df_without_smiles)} - SMILES Generation Error.csv")
    ligand_df_with_smiles = ligand_df[~ligand_df.canonique_smiles.str.contains("ERROR")]
    ligand_df_with_smiles.to_csv(f"{save_files_path}ligand_df_with_SMILES.csv")

    # %% [markdown]
    # ### Get <font color = "red"> info </font> for dataset processing

    # %%

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

        info = pd.DataFrame(info, columns = ["ligand_name", "pdb_code", "canonique_smiles", 
                                            "ligand_fpath", "ligand_ftype", "affinity", "pocket_name", "pocket_com", "release_year", "right_pocket_by_distance"])
        error_info = pd.DataFrame(error_info, columns = ["pdb_code", "info", "ligand_name", "ligand_fpath", "release_year"])
        qrint(f"Successfully processed {R}info{S}.")
        
        # Remove recorded bad info
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

    # %%
    info

    # %% [markdown]
    # ### Construct dataset

    # %%
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

    if True:
        class MyDataset_VS(Dataset):
            def __init__(self, root, df_tr=None, df_te=None, df_va=None, df_te2=None, protein_dict=None, proteinMode=0, compoundMode=1,
                        pocket_radius=20, shake_nodes=None, 
                        transform=None, pre_transform=None, pre_filter=None, generate_3D_conf = False,
                        protein_res_id_dict=None, nci_df=None, ligand_atom_id_dict=None, cfg_mode=None,
                        right_pocket_by_distance=True,
                        ):
                self.trainindx = 0
                self.testindx = 0
                self.valindx = 0
                self.test2indx = 0
                self.test_df = df_te
                self.train_df = df_tr
                self.val_df = df_va
                self.test2_df = df_te2
                self.testindx = len(self.test_df) 
                self.expected_test_count = len(self.test_df) 
                self.trainindx = len(self.train_df) +  self.testindx 
                self.expected_train_count = len(self.train_df) 
                self.valindx = len(self.val_df) + self.trainindx
                self.expected_val_count = len(self.val_df)
                self.test2indx = (len(self.test2_df) + self.valindx) if self.test2_df is not None else 0

                self.expected_test2_count = len(self.test2_df) if self.test2_df is not None else 0 
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
                _data_total = pd.concat([self.test_df, self.train_df, self.val_df, self.test2_df]) 
                torch.save(_data_total, self.processed_paths[0])
                torch.save(self.protein_dict, self.processed_paths[1])

            def len(self):
                return self.expected_test_count + self.expected_train_count \
                    + self.expected_val_count + self.expected_test2_count

            def get_idx_split(self):
                return self.expected_test_count, \
                    self.expected_test_count + self.expected_train_count, \
                    self.expected_test_count + self.expected_train_count + self.expected_val_count, \
                    self.expected_test_count + self.expected_train_count + self.expected_val_count + self.expected_test2_count

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

    # %% [markdown]
    # #### data split

    # %% [markdown]
    # ### datasets generation or load

    # %%
    cfg_use_saved_files = False
    cfg_save_files=True

    # %%
    save_files_path

    # %%
    if True:
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
                    for _s in ["data"]:
                        os.system(f"mkdir -p {dataset_path}{_s}")
                info_test2 = info[~(info.pdb_code.isin(_train)|info.pdb_code.isin(_val)|info.pdb_code.isin(_test))]
                _total_set = MyDataset_VS(f"{dataset_path}data/", df_tr=info[info.pdb_code.isin(_train)],df_te=info[info.pdb_code.isin(_test)],df_va=info[info.pdb_code.isin(_val)], df_te2=info_test2, \
                                        protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode)
                _te, _tr, _va, _te2 = _total_set.get_idx_split()
                test_set = _total_set[:_te]
                train_set = _total_set[_te:_tr]
                val_set = _total_set[_tr:_va]
                test2_set = _total_set[_va:_te2]
                    
        if not cfg_timesplit:
            test2_set = None
            train_part, val_part, test_part = cfg_split_strategy[0], cfg_split_strategy[1], cfg_split_strategy[2]
            train_val_split, val_test_split = int(train_part * len(info)), int((train_part+val_part) * len(info))
            qrint(f"{R}Length% split strategy{S}: {train_val_split} - {val_test_split-train_val_split} - {len(info)-val_test_split}. For dataset processing, the results will always be saved.")
            dataset_path = f"{save_files_path}dataset/"
            qrint(f"Processing {R}dataset{S}:")
            if not cfg_use_saved_files:
                os.system(f"rm -r {dataset_path}")
                for _s in ["data"]:
                    os.system(f"mkdir -p {dataset_path}{_s}")
            _total_set = MyDataset_VS(f"{dataset_path}data/", df_tr=info.iloc[0:train_val_split], df_te=info.iloc[val_test_split:], df_va=info.iloc[train_val_split:val_test_split], df_te2=None, \
                                        protein_dict=protein_dict, protein_res_id_dict=protein_res_id_dict, nci_df=nci_df, ligand_atom_id_dict=ligand_atom_id_dict, cfg_mode=cfg_mode)
            _te, _tr, _va, _te2 = _total_set.get_idx_split()
            test_set = _total_set[:_te]
            train_set = _total_set[_te:_tr]
            val_set = _total_set[_tr:_va]
            test2_set = _total_set[_va:_te2]
            
        qrint(f"Successfully processed {R}dataset{S}s.")
        qrint(f"-- {R}train set{S} : {len(train_set)}")
        qrint(f"-- {R}val set{S}   : {len(val_set)}")
        qrint(f"-- {R}test set{S}  : {len(test_set)}")
        qrint(f"-- {R}test2 set{S} : {len(test2_set) if test2_set else 0}")

    # %%
    def checkdata(data, fpath = "./trash.txt", cfg_mode=cfg_mode):
        if isinstance(data, str) or isinstance(data, list) or data is None:
            with open(fpath, "a") as f:
                f.write(f"{data} error!\n")        
        elif cfg_mode=="nciyes":
            device = data.y_batch.device
            samples = []
            shapes = []
            ff = True

            # check nci record in right_pocket
            for i, (_right, _shape) in enumerate(zip(data.right_pocket_by_distance, data.pair_shape)):
                if _right == True:
                    samples.append(i)
                    shapes.append(_shape)
            samples = torch.tensor(samples).to(device)
            index = torch.isin(data.y_batch, samples)
            ss = data.nci_sequence[index]
            if len(samples) and not (ss.sum()):
                with open(fpath, "a") as f:
                    f.write(f"{data.dataname} : its right pocket has no NCI record!\n")
                    
    def checkdata2(data, fpath = "./trash2.txt", cfg_mode=cfg_mode):
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

    # %%
    cfg_checkdata=True

    # %% [markdown]
    # ## Training Block

    # %%
    import logging
    from torch_geometric.loader import DataLoader

    # %%
    device = f"cuda:{args.gpu_id}"
    CUDA_LAUNCH_BLOCKING=1
    if True:
        if cfg_mode == "tankbind":
            from model import get_model
            model = get_model(0, logging, device)
            modelFile = "../saved_models/self_dock.pt"
            model.load_state_dict(torch.load(modelFile, map_location=device))

        elif cfg_mode == "nciyes":
            from sx_model import sx_get_model
            model = sx_get_model(0, logging, device, nciyes=True, margin=1, margin_weight=1, nci_weight=0, output_classes=2,
                                class_weight=torch.tensor([1,1],dtype=torch.float32))
            IaBNetFile = "../saved_models/self_dock.pt"
            model.IaBNet.load_state_dict(torch.load(IaBNetFile, map_location=device))

        elif cfg_mode == "frag":
            from model_frag import get_model
            model = get_model(0, logging, device)
            modelFile = "../saved_models/self_dock.pt" #?
            model.load_state_dict(torch.load(modelFile, map_location=device))#?

        logging.basicConfig(level=logging.INFO)
        _ = model.eval()


    # %%

        
        



    # %%
    from sx_new_class import InstanceDynamicBatchSampler
    mx = 1000
    logging.basicConfig(level=logging.INFO)
    if cfg_running_mode == "train":
        train_data_loader = DataLoader(train_set, batch_sampler = InstanceDynamicBatchSampler(dataset=train_set, max_num=mx, mode="node", shuffle=True), follow_batch=['x', 'y', 'compound_pair'], num_workers=3)
        val_data_loader = DataLoader(val_set, batch_sampler = InstanceDynamicBatchSampler(dataset=val_set, max_num=mx, mode="node", shuffle=True), follow_batch=['x', 'y', 'compound_pair'], num_workers=3) if len(val_set) else None
        test_data_loader = DataLoader(test_set, batch_sampler = InstanceDynamicBatchSampler(dataset=test_set, max_num=mx, mode="node", shuffle=True), follow_batch=['x', 'y', 'compound_pair'], num_workers=1) if len(test_set) else None
        test2_data_loader = DataLoader(test2_set,batch_sampler = InstanceDynamicBatchSampler(dataset=test2_set, max_num=mx, mode="node", shuffle=True), follow_batch=['x', 'y', 'compound_pair'], num_workers=1) if test2_set else None
    else:
        test_data_loader = DataLoader(test_set, batch_sampler = InstanceDynamicBatchSampler(dataset=test_set, max_num=mx, mode="node", shuffle=True), follow_batch=['x', 'y', 'compound_pair'], num_workers=1) if len(test_set) else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    list_val_metric = []

    #from Collections import defaultdict

    nci_dict = {}

    # %%

    if cfg_running_mode == "train":
        for _epoch in range(args.max_epoch):
            model.train()
            _list_loss = []
            _list_nci = []
            _list_score = []
            errors, error_file = [], f"{main_path}train_{_epoch}_errors.txt"
            for i,data in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
                if cfg_checkdata:
                    checkdata(data)
                    if isinstance(data, str) or isinstance(data, list):
                        errors.append(data)
                        with open(error_file, "a") as f:
                            f.write("\n"+str(data)+"\n")    
                        continue
                data = data.to(device)
                if cfg_mode == "tankbind": # Results : affinity_pred_dictbb
                    y_pred, affinity_pred = model(data)
                elif cfg_mode == "nciyes": # Results : many
                    data = data.to(device)
                    y_pred, affinity_pred, nci_pred = model(data, i)
                    loss = model.calculate_loss(affinity_pred, y_pred, nci_pred, data.affinity, data.dis_map,
                                                data.nci_sequence, data.right_pocket_by_distance, i, data.y_batch, data.pair_shape)
                    score = model.calculate_aff_score(affinity_pred, data.affinity).detach().cpu()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    _list_loss.append(loss.detach().cpu())
                    _list_nci.append(nci_pred.detach().cpu())
                    _list_score.append(score)
                    writer.add_scalar(f'BatchLoss/train_{_epoch}', loss.detach().cpu()/len(data), i)
                    writer.add_scalar(f'BatchScore/train_{_epoch}', score/len(data), i)
                    
            nci_dict[f"train_{_epoch}"] = _list_nci
            
            losst = torch.mean(torch.cat([torch.tensor([i]) for i in _list_loss]), dim=0)
            scoret = torch.mean(torch.cat([torch.tensor([i]) for i in _list_score]), dim=0)
            with open (f"{main_path}/epoch_{_epoch}_scores_tr.pkl", "wb") as f:
                pickle.dump(scoret, f)
            qrint('TRAIN----> Epoch [{}/{}], Loss: {:.4f}' .format(_epoch+1, args.max_epoch, losst.item()))
            qrint('TRAIN----> Epoch [{}/{}], Score: {:.4f}' .format(_epoch+1, args.max_epoch, scoret.item()))
            writer.add_scalar('Loss/train', losst.item(), _epoch)
            writer.add_scalar('Score/train', scoret.item(), _epoch)
            model.eval()
            
            if val_data_loader is not None:
                with torch.no_grad():
                    
                    _list_loss_va = []
                    _list_nci = []
                    _list_score_va = []
                    for i, data in tqdm(enumerate(val_data_loader), total=len(val_data_loader)):
                        data = data.to(device)
                        if cfg_mode == "tankbind": # Results : affinity_pred_dict
                            y_pred, affinity_pred = model(data)
                        elif cfg_mode == "nciyes": # Results : many
                            y_pred, affinity_pred, nci_pred = model(data, i)
                            loss = model.calculate_loss(affinity_pred, y_pred, nci_pred, data.affinity, data.dis_map,
                                                        data.nci_sequence, data.right_pocket_by_distance, i, data.y_batch, data.pair_shape)
                            score = model.calculate_aff_score(affinity_pred, data.affinity).detach().cpu()
                            _list_loss_va.append(loss)
                            _list_nci.append(nci_pred)
                            _list_score_va.append(score)
                            writer.add_scalar(f'BatchLoss/valid_{_epoch}', loss.detach().cpu()/len(data), i)
                            writer.add_scalar(f'BatchScore/valid_{_epoch}', score/len(data), i)
                    losst_va = torch.mean(torch.cat([torch.tensor([i]) for i in _list_loss_va]), dim=0)
                    score_va = torch.mean(torch.cat([torch.tensor([i]) for i in _list_score_va]), dim=0)
                    
                    nci_dict[f"val_{_epoch}"] = _list_nci
                    qrint('VALID----> Epoch [{}/{}], Loss: {:.4f}' .format(_epoch+1, args.max_epoch, losst_va.item()))
                    qrint('VALID----> Epoch [{}/{}], Score: {:.4f}' .format(_epoch+1, args.max_epoch, score_va.item()))
                    writer.add_scalar('Loss/valid', losst_va.item(), _epoch)
                    writer.add_scalar('Score/valid', score_va.item(), _epoch)
                    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': _epoch}
                    model_dir = save_model_path + f'/{_dirname}-%s/lr%s-batchsize%s-%s' % (args.iter, args.lr, args.batch_size, args.model_mode)
                    if not os.path.exists(model_dir):
                        os.system(f"mkdir -p {model_dir}")
                    epoch_model_dir = f'%s/epoch-%s' % (model_dir, _epoch + 1)
                    torch.save(state, epoch_model_dir)
                    with open (f"{main_path}/epoch_{_epoch}_scores_va.pkl", "wb") as f:
                        pickle.dump(score_va, f)
            model.eval() 
            with torch.no_grad():
                ## TODO 定义一个score用来评估

                if test_data_loader is not None:
                    _list_loss_te = []
                    _list_nci = []
                    _list_score_te = []
                    for i, data in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
                        data = data.to(device)
                        if cfg_mode == "tankbind": # Results : affinity_pred_dict
                                data = data.to(device)
                                y_pred, affinity_pred = model(data)
                        elif cfg_mode == "nciyes": # Results : many
                            y_pred, affinity_pred, nci_pred = model(data, i)
                            loss = model.calculate_loss(affinity_pred, y_pred, nci_pred, data.affinity, data.dis_map,
                                                        data.nci_sequence, data.right_pocket_by_distance, i, data.y_batch, data.pair_shape)
                            score = model.calculate_aff_score(affinity_pred, data.affinity)
                            _list_loss_te.append(loss)
                            _list_nci.append(nci_pred)
                            _list_score_te.append(score)
                            writer.add_scalar(f'BatchLoss/test_{_epoch}', loss.detach().cpu()/len(data), i)
                            writer.add_scalar(f'BatchScore/test_{_epoch}', score/len(data), i)
                    nci_dict[f"test_{_epoch}"] = _list_nci
                    losst_te = torch.mean(torch.cat([torch.tensor([i]) for i in _list_loss_te]), dim=0)
                    score_te = torch.mean(torch.cat([torch.tensor([i]) for i in _list_score_te]), dim=0)
                    qrint('TEST----> loss: {}' .format(losst_te))
                    qrint('TEST----> Epoch [{}/{}], Score: {:.4f}' .format(_epoch+1, args.max_epoch, score_te.item()))
                    with open (f"{main_path}/epoch_{_epoch}_scores_test.pkl", "wb") as f:
                        pickle.dump(score_te, f)
                    writer.add_scalar('Loss/test', losst_te.item(), _epoch)
                    writer.add_scalar('Score/test', score_te.item(), _epoch)
            if test2_data_loader is not None:
                _list_loss_te_2 = []
                for i, data in tqdm(enumerate(test2_data_loader), total=len(test2_data_loader)):
                    if cfg_mode == "tankbind": # Results : affinity_pred_dict
                            data = data.to(device)
                            y_pred, affinity_pred = model(data)
                    elif cfg_mode == "nciyes": # Results : many
                        y_pred, affinity_pred, nci_pred = model(data, i)
                        loss = model.calculate_loss(affinity_pred, y_pred, nci_pred, data.affinity, data.dis_map,
                                                    data.nci_sequence, data.right_pocket_by_distance, i, data.y_batch, data.pair_shape)
                        score = model.calculate_aff_score(affinity_pred, data.affinity)
                        _list_loss_te_2.append(loss)
                        _list_nci.append(nci_pred)
                        _list_score_te.append(score)
                        writer.add_scalar(f'BatchLoss/test2_{_epoch}', loss.detach().cpu()/len(data), i)
                        writer.add_scalar(f'BatchScore/test2_{_epoch}', score/len(data), i)
                losst_te_2 = torch.mean(torch.cat([torch.tensor([i]) for i in _list_loss_te_2]), dim=0)
                score_te_2 = torch.mean(torch.cat([torch.tensor([i]) for i in _list_score_te]), dim=0)
                with open (f"{main_path}/epoch_{_epoch}_scores_test_2.pkl", "wb") as f:
                    pickle.dump(score_te_2, f)
                qrint('TEST2----> loss: {}' .format(losst_te_2))
                qrint('TEST2----> Epoch [{}/{}], Score: {:.4f}' .format(_epoch+1, args.max_epoch, score_te_2.item()))

    writer.close()


    # %% [markdown]
    # ## Inference

    # %%
    scores = []
    if cfg_running_mode == "inference":
        for _epoch in range(1):
            model.eval()
            with torch.no_grad():
                if test_data_loader is not None:
                    _list_loss_te = []
                    #_list_nci = []
                    _score = []
                    for data in tqdm(test_data_loader):
                        model.eval()
                        data = data.to(device)
                        if cfg_mode == "tankbind": # Results : affinity_pred_dict
                                data = data.to(device)
                                y_pred, affinity_pred = model(data)
                        elif cfg_mode == "nciyes": # Results : many
                            y_pred, affinity_pred, nci_pred = model(data, i)
                            loss = model.calculate_loss(affinity_pred, y_pred, nci_pred, data.affinity, data.dis_map,
                                                        data.nci_sequence, data.right_pocket_by_distance, i, data.y_batch, data.pair_shape)
                            score = model.calculate_aff_score(affinity_pred, data.affinity)
                            _list_loss_te.append(loss)
                            #_list_nci.append(nci_pred)
                        
                    
                    nci_dict[f"test_{_epoch}"] = _list_nci
                    losst_te = torch.cat([torch.tensor([i]) for i in _list_loss_te])
                    losst_te = torch.mean(losst_te, dim=0)
                    scores.append(score)
                    print('TEST----> loss: {}' .format(losst_te))
                    with open(f"{main_path}Scores", "wb") as f:
                        f.dump(scores, f)
                                        
    '''
                if test2_data_loader is not None:
                    _list_loss_te_2 = []
                    for data in tqdm(test2_data_loader):
                        if cfg_mode == "tankbind": # Results : affinity_pred_dict
                                data = data.to(device)
                                y_pred, affinity_pred = model(data)
                        elif cfg_mode == "nciyes": # Results : many
                            y_pred, affinity_pred, nci_pred = model(data, i)
                            loss = model.calculate_loss(affinity_pred, y_pred, nci_pred, data.affinity, data.dis_map,
                                                        data.nci_sequence, data.right_pocket_by_distance, i, data.y_batch, data.pair_shape)
                            _list_loss_te_2.append(loss)

                    losst_te_2 = torch.cat([torch.tensor([i]) for i in _list_loss_te_2])
                    losst_te_2 = torch.mean(losst_te_2, dim=0)
                    print('TEST2----> loss: {}' .format(losst_te_2))
    '''

    # %%
    type(train_set)

    # %% [markdown]
    # ### Results

    # %%
    '''
    import pickle
    if cfg_mode == "tankbind":
        pass
    elif cfg_mode == "nciyes":
        with open(f"{main_path}errors.pkl", "wb") as f:
            pickle.dump(errors, f)
        with open(f"{main_path}y_pred_dict.pkl", "wb") as f:
            pickle.dump(y_pred_dict, f)
        with open(f"{main_path}nci_pred_dict.pkl", "wb") as f:
            pickle.dump(nci_pred_dict, f)
        with open(f"{main_path}nci_true_dict.pkl", "wb") as f:
            pickle.dump(nci_true_dict, f)
        with open(f"{main_path}affinity_pred_dict.pkl", "wb") as f:
            pickle.dump(affinity_pred_dict, f)
        with open(f"{main_path}data_name_dict.pkl", "wb") as f:
            pickle.dump(data_name_dict, f)    
        with open(f"{main_path}dis_map_dict.pkl", "wb") as f:
            pickle.dump(dis_map_dict, f)
        with open(f"{main_path}loss_dict.pkl", "wb") as f:
            pickle.dump(loss_dict, f)
        with open(f"{main_path}right_pocket_dict.pkl", "wb") as f:
            pickle.dump(right_pocket_dict, f)
    elif cfg_mode == "frag":
        pass
        #TODO: write your codes here
    '''
    # %% [markdown]
    # |CODE|EPOCH|TP|TN|FP|FN|Preci.%|Recall%|Notes
    # |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
    # |NCIYes-test00-08150437|val_0|#25175|4490395|1414547|557|5.58|97.83|weights: 1, 500|
    # |NCIYes-test00-08150549|val_0|9325|2553748|398723|3541|22.85|72.47|weights: 1, 100|
    # |NCIYes-test00-08150616|val_0|0|2952471|0|12866|-|-|weights: 1, 50|
    # |NCIYes-test00-08150700|val_0|10390|2496756|455715|2476|22.29|80.75|weights: 1, 80|
    # |NCIYes-test00-08150700|val_0|18514|5197212|707730|7218|2.54|80.75|weights: 1, 80|

    # %%

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default='data')

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--config_mode", choices=['tankbind', "nciyes", "frag"], default='tankbind')
    parser.add_argument("--data_version", type=str, default='v8.26')    
    parser.add_argument("--first_run", type=bool, default=False) 
    parser.add_argument("--model_mode", choices=['init', 'Halfbind', 'Tankbind'], default='Tankbind')
    parser.add_argument("--jupyter", type=bool, default=False)
    args = parser.parse_args()
    
    main(args)