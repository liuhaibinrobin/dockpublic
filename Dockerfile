FROM harbor.stonewise.cn/kubeflow/liuhaibin/notebook-image:tankbinddev10
# RUN mkdir /PaddleHelix
# ADD PaddleHelix /PaddleHelix
WORKDIR /home/jovyan/TankBind/ours
# WORKDIR /PaddleHelix/apps/pretrained_compound/ChemRL/GEM
# WORKDIR /home/jovyan/PaddleHelix/apps/pretrained_compound/ChemRL/GEM
ENV PATH /opt/conda/envs/tankbind_py38/bin:$PATH
SHELL ["/opt/conda/condabin/conda", "run", "-n", "tankbind_py38", "/bin/bash", "--login", "-c"]
# ENTRYPOINT ["/root/anaconda3/condabin/conda", "run", "-n", "tankbind_py38", "/bin/bash", "scripts/finetune_rank.sh"]