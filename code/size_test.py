import numpy as np
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, ignore_done=True); env.reset()
base = env.env if hasattr(env, "env") else env; sim = base.sim
slot = base.objects_dict["charging_slot_1"]; bid = sim.model.body_name2id(slot.root_body)
R = sim.data.body_xmat[bid].reshape(3,3); p = sim.data.body_xpos[bid]

walls, bases = [], []
for gid in range(sim.model.ngeom):
    n = sim.model.geom_id2name(gid) or ""
    if not n.startswith("charging_slot_1_") or n.endswith("_vis"): continue
    c = R.T @ (sim.data.geom_xpos[gid] - p)
    Rg = R.T @ sim.data.geom_xmat[gid].reshape(3,3)
    h = np.abs(Rg) @ sim.model.geom_size[gid]  # 在slot坐标系下xyz半尺寸
    item = (n, c, h)
    (bases if "base" in n else walls).append(item)

xwalls = [w for w in walls if w[2][1] > w[2][0]]
ywalls = [w for w in walls if w[2][0] >= w[2][1]]
L, Rw = min(xwalls, key=lambda t: t[1][0]), max(xwalls, key=lambda t: t[1][0])
B, T  = min(ywalls, key=lambda t: t[1][1]), max(ywalls, key=lambda t: t[1][1])

inner_w = abs((Rw[1][0]-Rw[2][0]) - (L[1][0]+L[2][0]))
inner_d = abs((T[1][1]-T[2][1]) - (B[1][1]+B[2][1]))
rim_z = min((w[1][2] + w[2][2]) for w in walls)      # 最低“槽口上沿”
base_top_z = max((b[1][2] + b[2][2]) for b in bases) # 最高“底板上表面”
inner_len = rim_z - base_top_z                         # 沿插入方向净长度

print(f"hole net width  (mm): {inner_w*1000:.3f}")
print(f"hole net depth  (mm): {inner_d*1000:.3f}")
print(f"hole net length (mm): {inner_len*1000:.3f}")
env.close()