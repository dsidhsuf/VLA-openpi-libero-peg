import numpy as np

from earbud_benchmark_v1 import get_task_specs, build_env


def geom_type_to_name(t):
    # mujoco geom types 常见映射
    mapping = {
        0: "plane",
        1: "hfield",
        2: "sphere",
        3: "capsule",
        4: "ellipsoid",
        5: "cylinder",
        6: "box",
        7: "mesh",
    }
    return mapping.get(int(t), f"unknown_{t}")


def pretty_size(geom_type_name, size):
    size = np.asarray(size).reshape(-1)
    if geom_type_name == "box":
        # mujoco box 的 size 是 half-extent
        full = 2.0 * size[:3]
        return {
            "half_extent_xyz": size[:3].tolist(),
            "full_extent_xyz": full.tolist(),
        }
    elif geom_type_name == "cylinder":
        # cylinder: [radius, half_length, ...]
        return {
            "radius": float(size[0]),
            "half_length": float(size[1]),
            "diameter": float(2 * size[0]),
            "full_length": float(2 * size[1]),
        }
    elif geom_type_name == "capsule":
        return {
            "radius": float(size[0]),
            "half_length": float(size[1]),
            "diameter": float(2 * size[0]),
            "full_length_without_caps": float(2 * size[1]),
        }
    else:
        return {
            "raw_size": size.tolist()
        }


def inspect_one_task(task):
    print("\n" + "=" * 80)
    print(f"task: {task.name}")
    print("=" * 80)

    env = build_env(task, camera_size=128)
    obs = env.reset()

    base = env.env if hasattr(env, "env") else env
    sim = base.sim
    model = sim.model

    geom_names = []
    for i in range(model.ngeom):
        name = model.geom_id2name(i)
        if name is None:
            continue
        geom_names.append((i, name))

    site_names = []
    for i in range(model.nsite):
        name = model.site_id2name(i)
        if name is None:
            continue
        site_names.append((i, name))

    print("\n[1] geoms containing 'earbud' / 'slot' / 'hole' / 'contain'")
    found_geom = False
    for i, name in geom_names:
        lname = name.lower()
        if any(k in lname for k in ["earbud", "slot", "hole", "contain"]):
            found_geom = True
            gtype = geom_type_to_name(model.geom_type[i])
            gsize = model.geom_size[i]
            print(f"\ngeom[{i}] {name}")
            print("  type:", gtype)
            print("  size:", pretty_size(gtype, gsize))
            print("  pos:", model.geom_pos[i].tolist())
    if not found_geom:
        print("  no matching geom found")

    print("\n[2] sites containing 'earbud' / 'slot' / 'hole' / 'contain'")
    found_site = False
    for i, name in site_names:
        lname = name.lower()
        if any(k in lname for k in ["earbud", "slot", "hole", "contain"]):
            found_site = True
            ssize = model.site_size[i]
            print(f"\nsite[{i}] {name}")
            print("  raw_size:", ssize.tolist())
            print("  pos:", model.site_pos[i].tolist())
    if not found_site:
        print("  no matching site found")

    print("\n[3] obs keys")
    print(list(obs.keys()))

    if "earbud_1_pos" in obs:
        print("\nobs earbud_1_pos:", np.asarray(obs["earbud_1_pos"]).tolist())
    if "charging_slot_1_pos" in obs:
        print("obs charging_slot_1_pos:", np.asarray(obs["charging_slot_1_pos"]).tolist())

    env.close()


def main():
    for task in get_task_specs():
        inspect_one_task(task)


if __name__ == "__main__":
    main()
