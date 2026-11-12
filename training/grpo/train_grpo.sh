#!/usr/bin/env bash
# SearchAgent-RL GRPO launcher, pinned to veRL 1876b06 (v0.9.1).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_ROOT="${VERL_ROOT:-/root/data/verl}"
PYTHON="${VERL_PYTHON:-${VERL_ROOT}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/data/models/Qwen3-4B-SFT-Merged}"
TRAIN_PATH="${TRAIN_PATH:-/root/data/datasets/grpo/nq_hotpotqa/verl/train.parquet}"
VAL_PATH="${VAL_PATH:-/root/data/datasets/grpo/nq_hotpotqa/verl/validation.parquet}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-/root/data/grpo/config/search_agent.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/grpo/checkpoints}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_grpo_v1}"

# Small run: one real optimizer step after a GPU instance is available.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-100}"
RESUME_MODE="${RESUME_MODE:-auto}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    TRAIN_BATCH_SIZE=2
    ROLLOUT_N=2
    TOTAL_TRAINING_STEPS=1
    TRAIN_MAX_SAMPLES=8
    VAL_MAX_SAMPLES=8
    SAVE_FREQ=-1
    TEST_FREQ=-1
    RESUME_MODE=disable
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_smoke"
fi
PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))

CHECK_CONFIG=0
if [[ "${1:-}" == "--check-config" ]]; then
    CHECK_CONFIG=1
    shift
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing veRL Python interpreter: $PYTHON (finish uv sync first)" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_PATH" || ! -f "$VAL_PATH" || ! -f "$AGENT_LOOP_CONFIG" ]]; then
    echo "Missing train/validation Parquet or AgentLoop YAML; check paths." >&2
    exit 1
fi
if [[ "$CHECK_CONFIG" == 0 && ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Missing merged SFT model: $MODEL_PATH (merge on GPU before training)" >&2
    exit 1
fi

# Ray workers must import SearchAgent-RL modules and use the same .venv.
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT"
cd "$VERL_ROOT"

HYDRA_ARGS=(
    model_engine=dp
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "data.train_files=${TRAIN_PATH}"
    "data.val_files=${VAL_PATH}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
    "data.val_max_samples=${VAL_MAX_SAMPLES}"
    data.max_prompt_length=2048
    data.max_response_length=6144
    data.truncation=error
    data.return_raw_chat=True
    data.dataloader_num_workers=2
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=1e-6
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.ref.fsdp_config.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
    actor_rollout_ref.rollout.max_model_len=8192
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.enforce_eager=True
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.agent.default_agent_loop=search_agent
    actor_rollout_ref.rollout.agent.num_workers=2
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG}"
    "reward.custom_reward_function.path=${PROJECT_ROOT}/training/grpo/reward.py"
    reward.custom_reward_function.name=compute_score
    reward.num_workers=2
    'trainer.logger=["console"]'
    trainer.project_name=SearchAgent-RL
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    trainer.nnodes=1
    trainer.n_gpus_per_node=2
    trainer.total_epochs="${TOTAL_EPOCHS}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    trainer.val_before_train=False
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.resume_mode=${RESUME_MODE}"
    trainer.max_actor_ckpt_to_keep=1
    "trainer.default_local_dir=${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
    "ray_kwargs.ray_init.runtime_env.py_executable=${PYTHON}"
)
if [[ "$CHECK_CONFIG" == 1 ]]; then
    exec "$PYTHON" -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" --cfg job --resolve "$@"
fi
exec "$PYTHON" -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}" "$@"
