# VLA OpenPI LIBERO Peg-Insertion Benchmark

Custom LIBERO assets and benchmark configurations for a peg-insertion task:
grasp a thin rectangular peg, align it vertically, and insert it into a slot.

## VLA OpenPI LIBERO Peg-Insertion Assets

- `libero_custom_peg/third_party/libero/libero/libero/bddl_files/`: custom BDDL tasks
- `libero_custom_peg/third_party/libero/libero/libero/envs/objects/`: custom
  programmatic object definitions
- `libero_custom_peg/benchmark/`: reproducible benchmark configurations and initial states
- `release_manifest.txt`: complete list of collected asset files
- `code/`: custom task-building, dataset, training, policy-server and evaluation scripts
- `code_manifest.txt`: complete list of packaged custom scripts
- `thesis_artifacts/`: lightweight CSV, JSON and NumPy artifacts mapped to thesis experiments
- `THESIS_ARTIFACTS.md`: thesis artifact scope and chapter mapping
- `thesis_artifacts_manifest.txt`: complete list of packaged thesis artifacts
- 该资产文件借助于libero固有资产（可组合几何体）进行调整、组合，仅供参考。
 
## 脚本生成文件等代码文件


- 该资产文件借助于libero固有资产（可组合几何体）进行调整、组合，仅供参考。
- augment与create等文件皆是数据生成脚本
- 针对生成的脚本结果可以使用convert脚本进行对上述脚本生成代码生成的数据进行lerobot格式的转换
- build代码为根据数据初始状态创建benchmark
- eval为评估脚本；
- finetune为微调脚本，包括lora与全量,推荐使用openpi的huggingface上的微调命令
- 由于libero与lerobot搭建环境存在差异，推荐使用服务端（policy.py）-客户端(eval)解耦形式处理

## 运行命令

### libero 环境初始化
cd /root/autodl-tmp/openpi_earbud_proto
source examples/libero/.venv/bin/activate

unset DISPLAY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/root/autodl-tmp/openpi_earbud_proto/third_party/libero:$PYTHONPATH

### libero 数据生成脚本
for ITEM in "easy 0" "medium 1000" "hard 2000"; do
  set -- $ITEM
  LEVEL=$1
  SEED=$2
  python "$SCRIPT" \
    --level "$LEVEL" \
    --episodes 20 \
    --seed "$SEED" \
    --seed-step 1 \
    --out-root "$OUT_ROOT" \
    --camera-names "agentview,frontview,robot0_eye_in_hand" \
    --target-video-duration-sec 60 \
    --video-fps 0 \
    --playback-speed 1 \
    --max-video-frames 0 \
    --flat-rest-prob 0
done

### 评测时liebro环境下的server命令示例

cd /root/autodl-tmp/openpi_earbud_proto
python /root/autodl-tmp/openpi_earbud_proto/policy_server_pi0.py \
  --policy_path /root/autodl-tmp/openpi_earbud_proto/outputs_pi0_lora_low_lr/pi0_lora_llm_action_vision_20260430_143015/checkpoints/002000/pretrained_model \
  --tokenizer_path /root/autodl-tmp/openpi_earbud_proto/outputs_pi0_lora_low_lr/pi0_lora_llm_action_vision_20260430_143015/checkpoints/002000/pretrained_model \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda \
  --chunk_len 50	
  
### 评测时lerobot环境下的client命令示例
cd /root/autodl-tmp/openpi_earbud_proto
unset DISPLAY
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/root/autodl-tmp/openpi_earbud_proto/third_party/libero:$PYTHONPATH

python eval_pi_libero_client_upright20x3_record.py \
  --server http://127.0.0.1:8000 \
  --camera_size 512 \
  --benchmark_assets /root/autodl-tmp/openpi_earbud_proto/benchmark/earbud_benchmark_v1_upright_20x3 \
  --expected_per_level 20 \
  --output_json earbud_pi05_libero_base_new_results.json \
  --record_video \
  --video_dir /root/autodl-tmp/openpi_earbud_proto/videos_8000*32 \
  --video_cameras agentview \
  --video_fps 20


### lora微调命令示例
  
##### 评测时需要关闭部分lora默认配置

python finetune_pi0_lora_action_expert_better_data.py \
  --dataset-path /root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_better_data \
  --model-path /root/autodl-tmp/hf_models/pi0_libero_base \
  --tokenizer-path /root/autodl-tmp/cache/huggingface/google/paligemma-3b-pt-224 \
  --output-root /root/autodl-tmp/openpi_earbud_proto/outputs_lora_action_expert_100*50 \
  --dataset-repo-id local/earbud_better_data \
  --job-name pi0_lora_action_expert_better_data \
  --steps 100 \
  --save-freq 50 \
  --batch-size 32 \
  --num-workers 4 \
  --lora-rank 32 \
  --optimizer-lr 1e-5 \
  --scheduler-warmup-steps 100 \
  --scheduler-decay-steps 1000 \
  --scheduler-decay-lr 1e-5 \
  --chunk-size 50 \
  --n-action-steps 50 \
  --seed 42 \
  --force-build-processors true \
  --offline true

### 数据集转换命令
python /root/autodl-tmp/openpi_earbud_proto/convert_libero_tree_to_single_lerobot.py \
  --src-root /root/autodl-tmp/openpi_earbud_proto/libero_lerobot_better_dataset \
  --output-root /root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_better_data \
  --repo-id local/earbud_keyframe_single \
  --categories easy \
  --limit-episodes 1 \
  --workers 4 \
  --prefetch 2 \
  --force-overwrite \
  --skip-invalid

### 全量微调命令

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_FORCE_BUILD_PROCESSORS=1
export PI0_LOCAL_TOKENIZER_PATH=/root/autodl-tmp/cache/huggingface/google/paligemma-3b-pt-224
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --dataset.repo_id=local/earbud_better_data \
  --dataset.root=/root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_better_data \
  --dataset.revision=v3.0 \
  --policy.path=/root/autodl-tmp/hf_models/pi0_libero_base_use_peft_false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.use_peft=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.optimizer_lr=1e-5 \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=2000 \
  --policy.scheduler_decay_lr=1e-6 \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --output_dir=/root/autodl-tmp/openpi_earbud_proto/outputs_full_action_expert/earbud_better_single_sanity_3000*32 \
  --job_name=earbud_better_single_sanity_1000 \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --eval_freq=0 \
  --batch_size=32 \
  --num_workers=2 \
  --steps=3000 \
  --log_freq=10 \
  --save_freq=1000 \
  --seed=42 \
  --resume=false
  
## 联系

如果你需要原论文，或者更多更详细的资料，联系QQ：2687439586
