# 1. Quick setup (one-time)
cd /root
git clone git@github.com:ghassen-fatnassi/Deep-HSI-CASSI-Torch.git  # Or upload files manually
cd Deep-HSI-CASSI-Torch

# 2. Install dependencies (one-time)
apt update && apt install -y git git-lfs python3-pip python3-venv libopenexr-dev tmux
python3 -m venv /root/hsi_env
source /root/hsi_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install numpy scipy matplotlib imageio wandb tqdm OpenEXR 

# 3. Make scripts executable
chmod +x run_experiment.sh
chmod +x scripts/*.sh

# 4. Launch the complete pipeline
./run_experiment.sh