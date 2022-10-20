if True:
    for nci_warmup in [True, False]:
        for nci_weight in [0, 25, 50, 100]:
            for under_sampling_ratio in [4, 8, 16]:
                sh_name = {True:"warmup", False:"no-warmup"}[nci_warmup] + f"-x{nci_weight}-ratio{under_sampling_ratio}.sh"
                warmup = {True:" --relative_nci_warm_up True", False:""}[nci_warmup]
                with open(sh_name, "w") as f:
                    f.writelines(
                        [
                            "#!/bin/sh\n",
                            "\n",
                            "cd /home/jovyan/TankBind/tankbind\n",
                            f"python main.py -d 0 -m 0 --batch_size 5 --label frag_eval_val --addNoise 5 --use_equivalent_native_y_mask{warmup} --nci_weight {nci_weight} --under_sampling_ratio {under_sampling_ratio}"
                        ]
                    )