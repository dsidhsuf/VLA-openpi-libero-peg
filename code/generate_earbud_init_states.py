import os
import numpy as np

from earbud_benchmark_v1 import get_task_specs, build_env, get_state_qpos_qvel

OUT_DIR = "/root/autodl-tmp/openpi_earbud_proto/benchmark_assets/init_states"
N_STATES_PER_TASK = 10
CAMERA_SIZE = 512

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for task in get_task_specs():
        env = build_env(task, camera_size=CAMERA_SIZE)

        states = []
        for seed in range(N_STATES_PER_TASK):
            env.seed(seed)
            obs = env.reset()

            # 如果你的 random_wrist_align 逻辑是在 reset 之后额外做的，
            # 应该在这里插入同样的“初始随机化/初始校正”函数，再保存 state。
            # 如果随机性已经在 BDDL/env.reset() 里，这里保持不变即可。

            states.append(get_state_qpos_qvel(env))

        out_path = task.init_states_file
        np.save(out_path, np.array(states, dtype=object), allow_pickle=True)
        print(f"[saved] {task.name}: {out_path} ({len(states)} states)")

        env.close()

if __name__ == "__main__":
    main()
