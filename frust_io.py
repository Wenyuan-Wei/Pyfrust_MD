from __future__ import annotations
from typing import Sequence, Optional, Union, Literal, Dict
import zarr
from zarr.codecs import BytesCodec, ZstdCodec
import MDAnalysis as mda 
from MDAnalysis.coordinates.PDB import PDBWriter
import numpy as np
import tempfile
import shutil
from pathlib import Path
import csv
import gzip

import pyfrust 
import pdb_parser 

AA_DEFAULT_ORDER = "-ACDEFGHIKLMNPQRSTVWY"

# choose your compression level (0 fast, 3–7 smaller)
ZSTD_LEVEL = 3

serializer = BytesCodec(endian="little")
compressors = (ZstdCodec(level=ZSTD_LEVEL, checksum=False),)
filters = ()

FrustMode = Literal["single", "pairwise"]
CollapseMode = Literal["keep", "mean", "sum", "max", "min", "wt"]


def _run_frust(
    pdb_path: str,
    mode: FrustMode,
    validate: bool = False, **kwargs
):
    if mode == "single":
        return pyfrust.single_frust(pdb_path, validate=validate, **kwargs)
    elif mode == "pairwise":
        return pyfrust.pairwise_frust(pdb_path, validate=validate, **kwargs)
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'single' or 'pairwise'.")


def write_frustration_zarr(
    top: str,
    traj: str,
    out_zarr_dir: str,
    frame_begin: int = 0,
    frame_end: Optional[int] = None,
    stride: int = 1,
    sele: str = "protein",
    chunk: int = 100,
    mode: FrustMode = "single",
    validate: bool = False,
    dtype: np.dtype = np.float32, **kwargs
) -> None:
    """
    Writes either single-site frustration or pairwise frustration to a zarr store.

    mode="single":
        Z   : (n_frames, L, A)
        Zwt : (n_frames, L)

    mode="pairwise":
        Z   : (n_frames, L, L, A, A)
        Zwt : (n_frames, L, L)

    Additional kwargs are passed to the underlying pyfrust functions. 
    """
    u = mda.Universe(top, traj)
    prot = u.select_atoms(sele)

    # Decide which frames you will process
    frame_indices = list(range(frame_begin, frame_end if frame_end is not None else len(u.trajectory), stride))
    n_frames = len(frame_indices)
    if n_frames == 0:
        raise ValueError("No frames selected (check frame_begin/frame_end/stride).")
    frame_chunk = min(chunk, n_frames)

    first_frame = frame_indices[0]
    u.trajectory[first_frame]

    # Keep your chainID hack (segids -> chainIDs)
    prot.atoms.chainIDs = np.array([s[-1] if s else "A" for s in prot.atoms.segids], dtype=object)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # --- probe first frame to learn shapes ---
        pdb_path0 = tmpdir / f"frame_{first_frame:06d}.pdb"
        with PDBWriter(str(pdb_path0)) as W:
            W.write(prot)

        sequence = pdb_parser.get_sequence(pdb_path0, chain=kwargs.get("chain", None))

        Z0, Zwt0 = _run_frust(str(pdb_path0), mode=mode, validate=validate, **kwargs)

        # Basic shape checks + infer L/A
        if mode == "single":
            if Z0.ndim != 2 or Zwt0.ndim != 1:
                raise ValueError(f"[single] Expected Z (L,A) and Zwt (L,), got {Z0.shape} and {Zwt0.shape}")
            L, A = Z0.shape
            Z_shape = (n_frames, L, A)
            Z_chunks = (frame_chunk, L, A)
            Zwt_shape = (n_frames, L)
            Zwt_chunks = (frame_chunk, L)

        else:  # mode == "pairwise"
            if Z0.ndim != 4 or Zwt0.ndim != 2:
                raise ValueError(f"[pairwise] Expected Z (L,L,A,A) and Zwt (L,L), got {Z0.shape} and {Zwt0.shape}")
            L1, L2, A1, A2 = Z0.shape
            if L1 != L2:
                raise ValueError(f"[pairwise] Expected square LxL for Z; got {Z0.shape}")
            if A1 != A2:
                raise ValueError(f"[pairwise] Expected square AxA for Z; got {Z0.shape}")
            if Zwt0.shape != (L1, L1):
                raise ValueError(f"[pairwise] Expected Zwt (L,L) matching Z; got {Zwt0.shape} vs L={L1}")
            L, A = L1, A1

            ij=32
            Z_shape = (n_frames, L, L, A, A)
            # Chunk along frames; keep inner dims whole (simple + safe)
            Z_chunks = (frame_chunk, ij, ij, A, A)
            Zwt_shape = (n_frames, L, L)
            Zwt_chunks = (frame_chunk, ij, ij)

        # --- create zarr store ---
        root = zarr.open_group(out_zarr_dir, mode="w")
        root.attrs["mode"] = mode
        root.attrs["top"] = str(top)
        root.attrs["traj"] = str(traj)
        root.attrs["sele"] = str(sele)
        root.attrs["stride"] = int(stride)
        root.attrs["sequence"] = str(sequence)
        root.attrs["aa_order"] = "-ACDEFGHIKLMNPQRSTVWY"

        Z = root.create_array(
            "Z", shape=Z_shape, chunks=Z_chunks, dtype=dtype,
            serializer=serializer, compressors=compressors, filters=filters,
        )
        Zwt = root.create_array(
            "Zwt", shape=Zwt_shape, chunks=Zwt_chunks, dtype=dtype,
            serializer=serializer, compressors=compressors, filters=filters,
        )

        root.create_array(
            "frame_indices",
            data=np.asarray(frame_indices, dtype=np.int32),
            serializer=serializer, compressors=compressors, filters=filters,
        )

        # --- write first frame ---
        Z[0, ...] = np.asarray(Z0, dtype=dtype)
        Zwt[0, ...] = np.asarray(Zwt0, dtype=dtype)

        # --- remaining frames ---
        for out_i, fi in enumerate(frame_indices[1:], start=1):
            u.trajectory[fi]

            pdb_path = tmpdir / f"frame_{fi:06d}.pdb"
            with PDBWriter(str(pdb_path)) as W:
                W.write(prot)

            Zi, Zwti = _run_frust(str(pdb_path), mode=mode, validate=validate, **kwargs)

            # (optional) assert shapes remain consistent
            if Zi.shape != Z0.shape or Zwti.shape != Zwt0.shape:
                raise ValueError(
                    f"Shape changed at frame {fi}: "
                    f"expected {Z0.shape}/{Zwt0.shape}, got {Zi.shape}/{Zwti.shape}"
                )

            Z[out_i, ...] = np.asarray(Zi, dtype=dtype)
            Zwt[out_i, ...] = np.asarray(Zwti, dtype=dtype)

            if out_i % 50 == 0:
                print(f"Wrote {out_i}/{n_frames} frames ({mode})")

    print(f"Done. Zarr written to: {out_zarr_dir} (mode={mode})")


def merge_shards(
    shards_dir: str,
    out_zarr_dir: str,
    delete_sources: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Merges shards produced by write_frustration_zarr, for either mode.

    Assumes every shard contains:
      - Z
      - Zwt
      - frame_indices
      - attrs['mode'] (optional, but recommended)
    """
    shards_dir = Path(shards_dir)
    out_zarr_dir = Path(out_zarr_dir)

    shard_paths = sorted([p for p in shards_dir.iterdir() if p.suffix == ".zarr" and p.is_dir()])
    if not shard_paths:
        raise FileNotFoundError(f"No .zarr shards found in {shards_dir}")

    shard_info = []
    for p in shard_paths:
        g = zarr.open_group(str(p), mode="r")
        fi = np.asarray(g["frame_indices"][:], dtype=np.int64)
        if fi.size == 0:
            continue
        shard_info.append((int(fi[0]), int(fi[-1]), fi.size, p))

    if not shard_info:
        raise ValueError("All shards had empty frame_indices.")

    shard_info.sort(key=lambda x: x[0])

    g0 = zarr.open_group(str(shard_info[0][3]), mode="r")
    mode0 = g0.attrs.get("mode", None)

    Z0 = g0["Z"]
    Zwt0 = g0["Zwt"]
    dtypeZ = Z0.dtype
    dtypeZwt = Zwt0.dtype
    Z_chunks   = Z0.chunks
    Zwt_chunks = Zwt0.chunks
    z_enc   = dict(serializer=Z0.serializer,   compressors=Z0.compressors,   filters=Z0.filters)
    zwt_enc = dict(serializer=Zwt0.serializer, compressors=Zwt0.compressors, filters=Zwt0.filters)


    total_frames = sum(n for _, _, n, _ in shard_info)

    if out_zarr_dir.exists():
        if overwrite:
            shutil.rmtree(out_zarr_dir)
            print(f"Overwriting existing {out_zarr_dir}...")
        else:
            raise FileExistsError(f"Output {out_zarr_dir} already exists.")

    out = zarr.open_group(str(out_zarr_dir), mode="w")
    if mode0 is not None:
        out.attrs["mode"] = mode0

    # Output shapes are (total_frames, ...) where "..." matches shard arrays
    Z_shape = (total_frames,) + tuple(Z0.shape[1:])
    Zwt_shape = (total_frames,) + tuple(Zwt0.shape[1:])

    

    Z_out = out.create_array("Z",   shape=Z_shape,   chunks=Z_chunks,   dtype=dtypeZ,   **z_enc)
    Zwt_out = out.create_array("Zwt", shape=Zwt_shape, chunks=Zwt_chunks, dtype=dtypeZwt, **zwt_enc)
    fi_chunk = 8192  # or 16384; tune if you like
    fi_out = out.create_array(
        "frame_indices",
        shape=(total_frames,),
        chunks=(min(fi_chunk, total_frames),),
        dtype=np.int32,
        serializer=Z0.serializer,
        compressors=Z0.compressors,
        filters=Z0.filters,
    )

    offset = 0
    for first_f, last_f, n, p in shard_info:
        g = zarr.open_group(str(p), mode="r")

        # Optional: enforce consistent mode across shards
        mode_i = g.attrs.get("mode", None)
        if mode0 is not None and mode_i is not None and mode_i != mode0:
            raise ValueError(f"Mode mismatch: first shard mode={mode0!r}, but {p.name} mode={mode_i!r}")

        Z = g["Z"]
        Zwt = g["Zwt"]
        fi = np.asarray(g["frame_indices"][:], dtype=np.int32)

        # Consistency checks
        if tuple(Z.shape[1:]) != tuple(Z0.shape[1:]):
            raise ValueError(f"Shape mismatch in {p}: Z is {Z.shape}, expected (*, {Z0.shape[1:]})")
        if tuple(Zwt.shape[1:]) != tuple(Zwt0.shape[1:]):
            raise ValueError(f"Shape mismatch in {p}: Zwt is {Zwt.shape}, expected (*, {Zwt0.shape[1:]})")
        if Z.shape[0] != n or Zwt.shape[0] != n or fi.shape[0] != n:
            raise ValueError(f"Length mismatch in {p}: Z/Zwt/frame_indices disagree")

        Z_out[offset : offset + n, ...] = Z[:]
        Zwt_out[offset : offset + n, ...] = Zwt[:]
        fi_out[offset : offset + n] = fi

        offset += n
        print(f"Merged {p.name}: frames {first_f}..{last_f} ({n})")

    # Carry over attrs
    try:
        out.attrs.update(g0.attrs.asdict() if hasattr(g0.attrs, "asdict") else dict(g0.attrs))
    except Exception:
        pass

    if delete_sources:
        for _, _, _, p in shard_info:
            shutil.rmtree(p)
        print("Deleted source shards.")

    print(f"Done. Merged store: {out_zarr_dir}")


def _kept_axes_from_collapse(mode: FrustMode, collapse_spec: Dict[str, CollapseMode]) -> list[str]:
    if mode == "single":
        canonical = ["frame", "L", "A"]
    else:
        canonical = ["frame", "L1", "L2", "A1", "A2"]
    return [ax for ax in canonical if collapse_spec.get(ax, "keep") == "keep"]

def _validate_aa_order(aa_order: str) -> None:
    if len(set(aa_order)) != len(aa_order):
        raise ValueError("aa_order contains duplicate characters.")
    if len(aa_order) == 0:
        raise ValueError("aa_order is empty.")

def _wt_index_array(seq: str, aa_order: str) -> np.ndarray:
    # map residue -> wt aa index
    mp = {aa: i for i, aa in enumerate(aa_order)}
    missing = sorted(set(seq) - set(mp))
    if missing:
        raise ValueError(f"seq contains characters not in aa_order: {missing}")
    return np.array([mp[a] for a in seq], dtype=np.int16)

def _letters_from_idx(idx: np.ndarray, aa_order: str) -> np.ndarray:
    # idx: int array
    if idx.size and int(idx.max()) >= len(aa_order):
        raise ValueError(f"AA index {int(idx.max())} out of range for aa_order length {len(aa_order)}.")
    # vectorized-ish mapping
    return np.fromiter((aa_order[int(i)] for i in idx.astype(np.int64, copy=False)),
                       dtype="<U1", count=idx.size)

def tensor_to_csv_rows(
    arr: np.ndarray,
    out_csv: Union[str, Path],
    *,
    # axis inference
    axis_names: Optional[Sequence[str]] = None,
    mode: Optional[FrustMode] = None,
    collapse_spec: Optional[Dict[str, CollapseMode]] = None,

    # For AA columns even when AA axes are collapsed:
    # Provide argmin/argmax choices for collapsed AA axes (required for min/max collapse).
    # Expected shapes:
    #   - single: aa_choice["A"]   has shape == arr.shape  (i.e., (F, L) if A collapsed)
    #   - pairwise: aa_choice["A1"] shape == arr.shape (if A1 collapsed)
    #              aa_choice["A2"] shape == arr.shape (if A2 collapsed)
    aa_choice: Optional[Dict[str, np.ndarray]] = None,

    # WT handling
    seq: Optional[str] = None,          # used when collapse_spec says wt on an AA axis
    aa_order: Optional[str] = None,     # if None -> AA_DEFAULT_ORDER

    # Header control
    headers: Optional[Dict[str, str]] = None,     # axis -> label (e.g., {"frame":"Frame","L1":"Res1"})
    value_header: str = "value",
    aa_idx_suffix: str = "_idx",
    aa_letter_suffix: str = "_aa",

    # NaN handling + safety
    drop_nan: bool = True,
    gzip_level: int = 6,
    max_rows_warn: int = 50_000_000,
    max_rows_abort: int = 300_000_000,
    float_fmt: str = "%.6g",
    chunk_outer: int = 1,
) -> None:
    """
    Stream a dense numpy tensor into a row-based CSV (optionally .gz).

    ALWAYS outputs AA columns:
      - single:  A_idx,  A_aa
      - pairwise: A1_idx, A1_aa, A2_idx, A2_aa

    How AA columns are computed:
      - If AA axis is present in axis_names: use the AA axis index from the grid.
      - If AA axis is collapsed with 'wt': compute WT AA index from seq + residue index (L/L1/L2).
      - If AA axis is collapsed with 'min'/'max': require aa_choice[...] with argmin/argmax indices.
        (Because the collapsed values alone cannot tell which AA achieved the min/max.)

    Notes:
      - This function assumes arr axis order matches the canonical order with removed axes
        (i.e., you removed axes but didn’t reorder the remaining ones).
      - For big outputs, prefer .csv.gz.
    """
    arr = np.asarray(arr)
    aa_order = AA_DEFAULT_ORDER if aa_order is None else str(aa_order)
    _validate_aa_order(aa_order)

    # Infer axis_names if needed
    if axis_names is None:
        if mode is None or collapse_spec is None:
            raise ValueError("Provide either axis_names=..., or both mode=... and collapse_spec=....")
        axis_names = _kept_axes_from_collapse(mode, collapse_spec)
    axis_names = list(axis_names)

    if arr.ndim != len(axis_names):
        raise ValueError(f"axis_names length ({len(axis_names)}) must match arr.ndim ({arr.ndim}).")

    # Determine which canonical AA axes we must output
    if mode is None:
        # If not provided, infer from names present in axis_names/collapse_spec
        # (best effort; recommended to provide mode)
        if any(ax in axis_names for ax in ("A1", "A2")) or (collapse_spec and any(k in collapse_spec for k in ("A1","A2"))):
            mode = "pairwise"
        else:
            mode = "single"

    if mode == "single":
        aa_axes_to_output = ["A"]
        residue_axes = ["L"]
    else:
        aa_axes_to_output = ["A1", "A2"]
        residue_axes = ["L1", "L2"]

    # Default headers
    default_headers = {
        "frame": "Frame",
        "L": "L",
        "L1": "L1",
        "L2": "L2",
        "A": "A",
        "A1": "A1",
        "A2": "A2",
    }
    if headers:
        default_headers.update(headers)

    # Build CSV header: axis columns (remaining axes), then AA idx/aa columns (always), then value
    header = [default_headers.get(ax, ax) for ax in axis_names]
    for ax in aa_axes_to_output:
        header.append(f"{default_headers.get(ax, ax)}{aa_idx_suffix}")
        header.append(f"{default_headers.get(ax, ax)}{aa_letter_suffix}")
    header.append(value_header)

    # Row estimate (upper bound before drop_nan)
    n_rows = int(np.prod(arr.shape)) if arr.size else 0
    if n_rows >= max_rows_abort:
        raise ValueError(
            f"Refusing to write {n_rows:,} rows (too large). "
            f"Slice/collapse further or increase max_rows_abort."
        )
    if n_rows >= max_rows_warn:
        print(f"Warning: writing up to {n_rows:,} rows (before drop_nan). Consider slicing or using .csv.gz.")

    out_csv = Path(out_csv)
    use_gz = out_csv.suffix == ".gz"
    opener = (lambda p: gzip.open(p, "wt", newline="", compresslevel=gzip_level)) if use_gz \
             else (lambda p: open(p, "w", newline=""))

    # Precompute WT indices if needed
    wt_full = None
    if collapse_spec is not None and "wt" in collapse_spec.values():
        if seq is None or not str(seq).strip():
            raise ValueError("WT AA columns requested via collapse_spec='wt' but seq is missing/empty.")
        wt_full = _wt_index_array(str(seq).strip(), aa_order)

    # Validate aa_choice shapes if provided
    aa_choice = aa_choice or {}
    for ax in aa_axes_to_output:
        if ax in aa_choice:
            choice_arr = np.asarray(aa_choice[ax])
            if choice_arr.shape != arr.shape:
                raise ValueError(
                    f"aa_choice['{ax}'] shape {choice_arr.shape} must match arr.shape {arr.shape} "
                    f"(it should be argmin/argmax indices aligned with the collapsed output)."
                )
            if not np.issubdtype(choice_arr.dtype, np.integer):
                raise ValueError(f"aa_choice['{ax}'] must be an integer array of AA indices.")

    def _need_choice_for(ax: str) -> bool:
        if collapse_spec is None:
            return False
        v = collapse_spec.get(ax, "keep")
        return v in ("min", "max")

    def _need_wt_for(ax: str) -> bool:
        if collapse_spec is None:
            return False
        return collapse_spec.get(ax, "keep") == "wt"

    # Identify position of residue axes in axis_names (if present)
    axis_pos = {ax: i for i, ax in enumerate(axis_names)}

    with opener(out_csv) as f:
        w = csv.writer(f)
        w.writerow(header)

        outer_size = arr.shape[0] if arr.ndim > 0 else 1

        for start in range(0, outer_size, chunk_outer):
            stop = min(outer_size, start + chunk_outer)
            block = arr[start:stop] if arr.ndim > 0 else arr  # (B,...)

            if block.ndim == 0:
                vals = block.reshape(-1)
                if drop_nan and not np.isfinite(vals)[0]:
                    continue
                # scalar: still must output AA columns — but we have no indices -> cannot
                raise ValueError("Cannot write AA columns for a scalar output (no residue/AA axes remain).")

            grids = np.indices(block.shape, dtype=np.int64)  # (ndim,B,d1,...)
            grids[0] += start  # globalize axis 0

            cols = [grids[i].reshape(-1) for i in range(block.ndim)]
            vals = block.reshape(-1)

            # Drop NaNs/inf in value column
            if drop_nan:
                m = np.isfinite(vals)
                if not m.any():
                    continue
                cols = [c[m] for c in cols]
                vals = vals[m]
            else:
                m = None  # no filtering

            # --- Build AA idx columns for each AA axis (always) ---
            aa_idx_cols = []
            aa_letter_cols = []

            for ax in aa_axes_to_output:
                # Case 1: AA axis is present in output -> take from index grid
                if ax in axis_pos:
                    aa_idx = cols[axis_pos[ax]].astype(np.int64, copy=False)

                else:
                    # AA axis was collapsed. Determine how.
                    if _need_wt_for(ax):
                        if wt_full is None:
                            raise ValueError(f"{ax} collapsed via 'wt' but WT indices could not be resolved.")
                        # Need the corresponding residue axis present to map residue -> WT AA
                        res_ax = "L" if ax == "A" else ("L1" if ax == "A1" else "L2")
                        if res_ax not in axis_pos:
                            raise ValueError(
                                f"Cannot output {ax} WT AA columns because residue axis {res_ax} "
                                f"is not present in output."
                            )
                        res_idx = cols[axis_pos[res_ax]].astype(np.int64, copy=False)
                        aa_idx = wt_full[res_idx].astype(np.int64, copy=False)

                    elif _need_choice_for(ax):
                        # Must have aa_choice with argmin/argmax indices aligned to arr
                        if ax not in aa_choice:
                            raise ValueError(
                                f"{ax} was collapsed with '{collapse_spec.get(ax)}', so AA identity is lost.\n"
                                f"To always output AA columns, pass aa_choice['{ax}'] containing arg{collapse_spec.get(ax)} "
                                f"indices aligned with the collapsed output (shape {arr.shape})."
                            )
                        # Slice the choice array same way as arr block and flatten, then apply the same drop_nan mask
                        choice_block = np.asarray(aa_choice[ax][start:stop])
                        choice_flat = choice_block.reshape(-1)
                        if drop_nan:
                            choice_flat = choice_flat[m]
                        aa_idx = choice_flat.astype(np.int64, copy=False)

                    else:
                        # Collapsed via something else (or unknown) -> cannot reconstruct
                        raise ValueError(
                            f"{ax} is not present in output and collapse_spec does not specify 'wt' or min/max for it; "
                            f"cannot output AA columns."
                        )

                aa_idx_cols.append(aa_idx)
                aa_letter_cols.append(_letters_from_idx(aa_idx, aa_order))

            # --- Write rows ---
            for row in zip(*cols, *sum(zip(aa_idx_cols, aa_letter_cols), ()), (float_fmt % v for v in vals)):
                w.writerow(row)