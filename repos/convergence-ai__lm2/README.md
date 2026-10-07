# LM2
[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

This is the official implementation of paper "LM2: Large Memory Models"

📝 [Arxiv](https://arxiv.org/abs/2502.06049v1) \|
🐱 [Code@GitHub](https://github.com/convergence-ai/lm2.git) \| 
🏠 [HomePage](https://convergence.ai/)
## Getting setup

Install the conda environment

```bash
conda env create -n lm2 -f environment.yaml
```

## Preparing your own datasets


### Command-Line Arguments

The preprocessing script accepts the following command-line arguments:

### Preprocessing Script Command-Line Arguments

| **Argument**         | **Flags**                | **Default Value**         | **Description**                                                                                                                                                          |
|----------------------|--------------------------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Version**          | `-v`, `--version`        | `"cosmo"`                 | Specifies the version (or subsample) of the SmolLM Corpus dataset to preprocess. Valid options: `cosmo`, `python`, `fineweb`, `smollm`.                                |
| **Model Descriptor** | `-m`, `--model_desc`     | `"llama-3"`               | Specifies the model descriptor to determine which tokenizer to use. Currently, only `"llama-3"` is supported (which uses the Meta Llama-3 tokenizer).                |
| **Shard Size**       | `-s`, `--shard_size`     | `100000000` (10^8 tokens) | Sets the maximum number of tokens per output binary shard file. Adjust this value based on your available disk space and processing requirements.                      |


### Example Usage

Tokenize using default parameters:

```bash
python data_proc/smollm.py -v cosmo -m llama-3
```

### Dataset Directory Mapping

The script uses a mapping between the `version` and `model_desc` arguments to determine where to save the processed dataset and which remote dataset to use from Hugging Face. The mapping is as follows:

| **Version** | **Model Descriptor** | **Local Directory**    | **Remote Dataset Name** |
|-------------|----------------------|------------------------|-------------------------|
| `cosmo`     | `llama-3`            | `cosmo-llama3`         | `cosmopedia-v2`         |
| `python`    | `llama-3`            | `python-llama3`        | `python-edu`            |
| `fineweb`   | `llama-3`            | `fineweb-ddp-llama3`    | `fineweb-edu-dedup`      |
| `smollm`    | `llama-3`            | `smollm-llama3`        | `cosmopedia-v2`         |

- **Local Directory:** This directory is used to store the processed binary shard files. It is located within the repository's `datasets/` folder.
- **Remote Dataset Name:** This name is used to load the dataset from Hugging Face using the `load_dataset` function.

Adjust the mapping as necessary to suit different dataset versions or model descriptors.

## Model Training

### Command-Line Arguments (Hydra Overrides)

The training script is configured using Hydra parameters passed as command-line overrides. Below is a description of each parameter used in the script:

| **Hydra Parameter**       | **Description**                                                                                                                                                     |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **model**                 | Specifies the model architecture to be used for training.                                                                                                         |
| **pretrain**              | Selects the default pretraining configuration.                                                                                                                    |
| **input_bin**             | Glob pattern indicating the location of the training data shards (binary files).                                                                                  |
| **input_val_bin**         | Glob pattern indicating the location of the validation data shards (binary files).                                                                                |
| **model.sequence_length** | Sets the sequence length (i.e., the number of tokens per sample) for training.                                                                                     |
| **model.use_memory**      | Boolean flag that enables the use of memory features in the model during training.                                                                                 |
| **train.batch_size**      | Specifies the batch size to be used during training.                                                                                                              |
| **train.dtype**           | Defines the data type used during training to optimize performance and memory usage.                                                                               |
| **train.learning_rate**   | The initial learning rate for the training optimizer.                                                                                                             |
| **train.warmup_iters**    | The number of iterations to gradually ramp up the learning rate during warmup.                                                                                      |
| **train.lr_decay_frac**   | Specifies the fraction of learning rate decay. A value of `0.0` indicates no decay is applied.                                                                       |
| **train.max_iters**       | The maximum number of training iterations to perform.                                                                                                             |
| **train.log_freq**        | Frequency (in iterations) at which training logs are output.                                                                                                      |
| **train.save_freq**       | Frequency (in iterations) at which model checkpoints are saved.

### Example Script
```bash
sh scripts/train.sh
```

## Code structure

```plaintext
LMLM/
├── configs/                    # Hydra configs
├── scripts/                    # Scripts for running experiments
├── data_proc/                    
│   ├── smollm.py               # Data preparation 
│   ├── data_common.py          # Utils for data preparation 
├── src/
│   ├── dataloader.py           # Loading preprocessed data
│   ├── memory.py               # Memory PyTorch Modules
│   ├── model_memory_llama.py   # Llama model with integrated memory module
│   ├── README.md               # Graph of Memory module structures
│   ├── trainer.py              # Training class, handling model training and inference loop
│   ├── utils.py                # Utility functions used across the project
├── .gitignore
├── train.py                    # Main training script
