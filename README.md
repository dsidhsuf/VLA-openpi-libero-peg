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
- 

## 联系

如果你需要原论文，或者更多更详细的资料，联系QQ：2687439586
