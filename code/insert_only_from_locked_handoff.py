#!/usr/bin/env python3
"""
Insertion-only entrypoint for fast tuning.

This wrapper reuses the full pipeline implementation but forces
`insert_fast_start=True` so grasp + upright alignment are bootstrapped
to a locked near-insert handoff state every run.
"""

import full_chain_pick_flat_lay_grasp_realign_retry_insert as full_chain


def build_parser():
    parser = full_chain.build_parser()
    parser.set_defaults(
        mode="single",
        level="easy",
        seed=0,
        grasp_attempts=6,
        insert_attempts=6,
        secondary_camera_name="overhead",
        secondary_camera_zoom=1.1,
        insert_fast_start=True,
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Always force insertion-only bootstrapping in this entrypoint.
    args.insert_fast_start = True
    print("insert_only_mode=locked_handoff")
    print("insert_fast_start=True")

    if args.mode == "sweep":
        full_chain.run_sweep(args)
    else:
        full_chain.run_single(args, args.seed)


if __name__ == "__main__":
    main()
