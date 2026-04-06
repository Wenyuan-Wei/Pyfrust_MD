"""
demo.py — End-to-end frustration pipeline demo

Usage
-----
    # Process all frames, split into shards of 50 frames each:
    python demo.py --topology path/to/protein.pdb \
                   --trajectory path/to/traj.xtc \
                   --chain A \
                   --selection "protein and segid A" \
                   --mode pairwise \
                   --shard-size 50

    # Process frames 0–100, no sharding:
    python demo.py --topology path/to/protein.pdb \
                   --trajectory path/to/traj.xtc \
                   --frame-end 100

    # Process all frames in one shot (no sharding):
    python demo.py --topology path/to/protein.pdb \
                   --trajectory path/to/traj.xtc

Steps performed
---------------
1. Compute frustration frame-by-frame and write to Zarr.
   - With --shard-size N: writes one shard per N-frame chunk, then merges.
   - Without --shard-size:  writes a single Zarr store directly.
2. Collapse the tensor (wildtype amino acids only, all frames/residues kept).
3. Export the collapsed result to CSV.

Output
------
  <out-dir>/frust.zarr        — merged (or direct) frustration tensor
  <out-dir>/frust.csv         — collapsed wildtype Z-scores
  <out-dir>/shards/           — intermediate shards (deleted after merge)
"""

import argparse
import math
from pathlib import Path
import numpy as np
import MDAnalysis as mda

import frust_io
import tensor_access
import pdb_parser


def parse_args():
    p = argparse.ArgumentParser(
        description="Pyfrust_MD demo: compute and export mutational frustration from an MD trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--topology",    required=True,  help="Path to topology file (PDB or equivalent).")
    p.add_argument("--trajectory",  required=True,  help="Path to trajectory file (XTC, DCD, etc.).")
    p.add_argument("--chain",       default="A",    help="Protein chain ID to analyse.")
    p.add_argument("--selection",   default="protein", help="MDAnalysis atom selection string.")
    p.add_argument("--mode",        default="pairwise", choices=["single", "pairwise"],
                   help="Frustration mode: 'single' (per-residue) or 'pairwise' (residue pairs).")
    p.add_argument("--frame-begin", type=int, default=0,
                   help="First trajectory frame to process (0-indexed).")
    p.add_argument("--frame-end",   type=int, default=None,
                   help="Last trajectory frame (exclusive). Omit or set to -1 to process all frames.")
    p.add_argument("--stride",      type=int, default=1,
                   help="Process every N-th frame.")
    p.add_argument("--shard-size",  type=int, default=None,
                   help="Frames per shard. If set, the trajectory is split into chunks of this "
                        "size, each written as a separate Zarr shard, then merged. "
                        "Omit to write everything in one shot.")
    p.add_argument("--out-dir",     default="./demo_output",
                   help="Directory for output files.")
    p.add_argument("--validate",    action="store_true",
                   help="Cross-check Z-scores against FrustratometeR (slow; for debugging).")
    return p.parse_args()


def _resolve_frame_end(topology: str, trajectory: str, frame_end: int | None) -> int:
    """Return the effective frame_end, loading the trajectory only when needed."""
    if frame_end is not None and frame_end != -1:
        return frame_end
    u = mda.Universe(topology, trajectory)
    return len(u.trajectory)


def _write_shards(args, frame_begin: int, frame_end: int, shards_dir: Path) -> None:
    """Split [frame_begin, frame_end) into chunks and write one shard each."""
    n_total = frame_end - frame_begin
    n_shards = math.ceil(n_total / args.shard_size)
    print(f"  Splitting {n_total} frames into {n_shards} shard(s) of ≤{args.shard_size} frames ...")

    for i in range(n_shards):
        shard_begin = frame_begin + i * args.shard_size
        shard_end   = min(shard_begin + args.shard_size, frame_end)
        shard_path  = shards_dir / f"frust_{shard_begin}_{shard_end}.zarr"

        print(f"  Shard {i+1}/{n_shards}: frames {shard_begin}–{shard_end} → {shard_path.name}")
        frust_io.write_frustration_zarr(
            top=args.topology,
            traj=args.trajectory,
            out_zarr_dir=str(shard_path),
            frame_begin=shard_begin,
            frame_end=shard_end,
            stride=args.stride,
            sele=args.selection,
            mode=args.mode,
            validate=args.validate,
            dtype=np.float32,
            chain=args.chain,
        )


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_path   = out_dir / "frust.zarr"
    csv_path    = out_dir / "frust.csv"
    shards_dir  = out_dir / "shards"

    frame_end = _resolve_frame_end(args.topology, args.trajectory, args.frame_end)
    frame_range_str = f"{args.frame_begin}–{frame_end}" + ("" if args.stride == 1 else f" (stride {args.stride})")

#    # ------------------------------------------------------------------
#    # Step 1: Compute frustration and write to Zarr
#    # ------------------------------------------------------------------
#    if args.shard_size:
#        print(f"[1/3] Computing {args.mode} frustration — frames {frame_range_str}, sharded ...")
#        shards_dir.mkdir(exist_ok=True)
#        _write_shards(args, args.frame_begin, frame_end, shards_dir)
#
#        print(f"  Merging shards into {zarr_path} ...")
#        frust_io.merge_shards(
#            shards_dir=str(shards_dir),
#            out_zarr_dir=str(zarr_path),
#            delete_sources=True,
#            overwrite=True,
#        )
#    else:
#        print(f"[1/3] Computing {args.mode} frustration — frames {frame_range_str} ...")
#        frust_io.write_frustration_zarr(
#            top=args.topology,
#            traj=args.trajectory,
#            out_zarr_dir=str(zarr_path),
#            frame_begin=args.frame_begin,
#            frame_end=frame_end,
#            stride=args.stride,
#            sele=args.selection,
#            mode=args.mode,
#            validate=args.validate,
#            dtype=np.float32,
#            chain=args.chain,
#        )

    # ------------------------------------------------------------------
    # Step 2: Collapse the tensor
    #   Keep all frames and residue axes; reduce amino-acid axes to
    #   the wildtype identity only (A → wt, or A1/A2 → wt).
    # ------------------------------------------------------------------
    print("[2/3] Collapsing tensor to wildtype amino acids ...")

    chain_seq = pdb_parser.get_sequence(args.topology, chain=args.chain)

    if args.mode == "single":
        collapse_param = {"frame": "keep", "L": "keep", "A": "wt"}
    else:
        collapse_param = {"frame": "keep", "L1": "keep", "L2": "keep",
                          "A1": "wt", "A2": "wt"}

    out_arr = tensor_access.collapse_frustration(
        zarr_path=str(zarr_path),
        mode=args.mode,
        collapse=collapse_param,
        seq=chain_seq,
        return_kind="numpy",
    )

    # ------------------------------------------------------------------
    # Step 3: Export to CSV
    # ------------------------------------------------------------------
    print(f"[3/3] Writing CSV to {csv_path} ...")

    frust_io.tensor_to_csv_rows(
        out_arr,
        csv_path,
        mode=args.mode,
        collapse_spec=collapse_param,
        seq=chain_seq,
    )

    dim_label = "frame, residue" if args.mode == "single" else "frame, residue1, residue2"
    print(f"\nDone.")
    print(f"  Zarr store : {zarr_path}")
    print(f"  CSV output : {csv_path}")
    print(f"  Array shape: {out_arr.shape}  ({dim_label})")


if __name__ == "__main__":
    main()
