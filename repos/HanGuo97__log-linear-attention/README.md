# Log Linear Attention

![Figure](figs/recurrent.png)

## Setup

1. Clone the repository and its submodules:
```bash
git clone --recursive https://github.com/HanGuo97/log-linear-attention.git
cd log-linear-attention
```

2. Install the package and its dependencies:
```bash
pip install -e .
pip install -e flame/
pip install -r flame/3rdparty/torchtitan/requirements.txt
```

### Docker Installation (Optional)

We provide a `Dockerfile` for containerized setup. To use it:

```bash
# Build the Docker image
DOCKER_BUILDKIT=1 docker build \
    -t log-linear-attention \
    -f Dockerfile \
    .

# Run the container
docker run -ti \
    --gpus all \
    log-linear-attention \
    bash
```

## Data Preparation

1. Configure the data preprocessing:
   - Open `hattention/preprocess_data.py`
   - Modify the save path to your desired location

2. Run the preprocessing script:
```bash
python -m hattentions.preprocess_data
```

> [!NOTE]
> The data preprocessing step may take a while.

## Training

1. Navigate to the training framework:
```bash
cd flame/
```

2. Launch training with the following command:
```bash
bash ../scripts/train_flame.sh --name [NAME] --config [CONFIG] --seed [--ac]
```

- `NAME`: Name for the experiment and save path
- `CONFIG`: Name of the config file in `configs/flame/` (without .json extension)
- `--seed`: Create a seed checkpoint before training
- `--ac`: Optional flag to enable activation checkpointing

> [!NOTE]
> 1. Before training, modify the absolute file paths in `scripts/train_flame.sh` to match your setup
> 2. The first training step will compile Triton kernels, which may take tens of minutes

# Acknowledgement
Special thanks to Tianyuan Zhang, Jyo Pari, Adam Zweiger, and Yu Zhang for lots of help and discussions.