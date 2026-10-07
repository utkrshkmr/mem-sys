export WANDB_PROJECT="hattention"

AC_MODE="none"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      NAME="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --seed)
      SEED_CKPT=1
      shift
      ;;
    --ac)
      AC_MODE="selective"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Arguments:"
echo "NAME: ${NAME}"
echo "CONFIG: ${CONFIG}"
echo "SEED_CKPT: ${SEED_CKPT}"
echo "AC_MODE: ${AC_MODE}"

if [[ -z "${NAME}" ]] || [[ -z "${CONFIG}" ]]; then
  echo "Usage: $0 --name NAME --config CONFIG [--seed] [--ac]"
  exit 1
fi

if [[ -n "${SEED_CKPT}" ]]; then

echo "Creating seed checkpoint..."

NNODE=1 NGPU=1 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/${NAME} \
  --model.config /workspace/main/log-linear-attention/configs/flame/${CONFIG}.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --training.batch_size 4 \
  --training.seq_len 16384 \
  --training.context_len 16384 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 95368 \
  --training.dataset arrow \
  --training.dataset_split train \
  --activation_checkpoint.mode ${AC_MODE} \
  --activation_checkpoint.selective_ac_option 4 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 25 \
  --checkpoint.create_seed_checkpoint

fi

echo "Training..."

NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/${NAME} \
  --model.config /workspace/main/log-linear-attention/configs/flame/${CONFIG}.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --training.batch_size 4 \
  --training.seq_len 16384 \
  --training.context_len 16384 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 95368 \
  --training.dataset arrow \
  --training.dataset_split train \
  --activation_checkpoint.mode ${AC_MODE} \
  --activation_checkpoint.selective_ac_option 4 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 25
