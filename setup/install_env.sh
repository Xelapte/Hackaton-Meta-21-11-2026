#!/bin/bash
# À lancer sur le NŒUD DE LOGIN (il a internet), une seule fois pour toute l'équipe.
# Avant : remplissez DIR dans slurm/env.sh. Depuis la racine du dépôt :
#   bash setup/install_env.sh
# Charge les modules de slurm/env.sh (même Python que sur les nœuds GPU), puis crée le venv.
set -eo pipefail
source slurm/env.sh || exit 1
set -u
python3 --version

python3 -m venv "$DIR/env"
source "$DIR/env/bin/activate"
pip install --upgrade pip

# Une seule commande pour que pip choisisse des versions compatibles entre elles :
# vLLM impose sa version de PyTorch (et de torchvision).
pip install vllm torchvision transformers accelerate bitsandbytes peft \
    scikit-learn pandas matplotlib huggingface_hub

python -c "import torch, vllm, transformers; \
print('torch', torch.__version__, '| vllm', vllm.__version__, \
'| transformers', transformers.__version__)"
echo "Environnement prêt : source $DIR/env/bin/activate"
