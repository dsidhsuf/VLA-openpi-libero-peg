# VLA OpenPI LIBERO Peg-Insertion Assets

Custom LIBERO assets and benchmark configurations for a peg-insertion task:
grasp a thin rectangular peg, align it vertically, and insert it into a slot.

## Contents

- `libero_custom_peg/third_party/libero/libero/libero/bddl_files/`: custom BDDL tasks
- `libero_custom_peg/third_party/libero/libero/libero/envs/objects/`: custom
  programmatic object definitions
- `libero_custom_peg/benchmark/`: reproducible benchmark configurations and initial states
- `release_manifest.txt`: complete list of collected files

## Installation

Overlay the packaged files onto the corresponding paths of an existing LIBERO checkout.

## Notes

Model checkpoints, generated training datasets and evaluation videos are intentionally excluded.
