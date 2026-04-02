from typing import Literal, Optional, Dict, Any
import numpy as np
import zarr

CollapseMode = Literal["keep", "max", "min", "wt"]
FrustMode = Literal["single", "pairwise"]

AA_DEFAULT_ORDER = "-ACDEFGHIKLMNPQRSTVWY"
REDUCERS = {
    "max":  np.nanmax,
    "min":  np.nanmin,
}

def _normalize_collapse(mode: FrustMode, collapse: Optional[Dict[str, CollapseMode]], collapse_residues: Optional[CollapseMode], collapse_amino_acids: Optional[CollapseMode]) -> Dict[str, CollapseMode]:
    if mode == "single":
        spec = {"frame": "keep", "L": "keep", "A": "keep"}
    else:
        spec = {"frame": "keep", "L1": "keep", "L2": "keep",
                "A1": "keep", "A2": "keep"}

    # Apply convenience defaults
    if collapse_residues is not None:
        if mode == "single":
            spec["L"] = collapse_residues
        else:
            spec["L1"] = collapse_residues
            spec["L2"] = collapse_residues

    if collapse_amino_acids is not None:
        if mode == "single":
            spec["A"] = collapse_amino_acids
        else:
            spec["A1"] = collapse_amino_acids
            spec["A2"] = collapse_amino_acids

    # Override with explicit collapse dict
    if collapse is not None:
        for k, v in collapse.items():
            if k not in spec:
                raise ValueError(f"Invalid collapse axis: {k}")
            spec[k] = v

    return spec

def _aa_to_index_map(aa_order: str) -> Dict[str, int]:
    if len(set(aa_order)) != len(aa_order):
        raise ValueError("aa_order contains duplicate characters.")
    return {aa: i for i, aa in enumerate(aa_order)}

def _wt_indices(seq: str, aa_order: str) -> np.ndarray:
    mp = _aa_to_index_map(aa_order)
    missing = sorted(set(seq) - set(mp))
    if missing:
        raise ValueError(f"seq contains characters not in aa_order: {missing}")
    return np.array([mp[a] for a in seq], dtype=np.int16)

def _as_indexer(x):
    # allow None, slice, list[int]
    return x if x is not None else slice(None)

def _apply_reduce(arr: np.ndarray, axis: int, mode: str) -> np.ndarray:
    # mode in {"mean","sum","max","min"}
    fn = REDUCERS[mode]
    return fn(arr, axis=axis)

def _canon_order_single():
    # canonical axis names for single: (F, L, A)
    return ["frame", "L", "A"]

def _canon_order_pairwise():
    # canonical axis names: (F, L1, L2, A1, A2)
    return ["frame", "L1", "L2", "A1", "A2"]

def _collapse_single(
    Z: "zarr.Array",
    collapse_spec: Dict[str, str],
    *,
    frames=None,
    residues=None,
    seq=None,
    aa_order=None,
) -> np.ndarray:
    # Slice from zarr (lazy until .__getitem__)
    Zs = Z[_as_indexer(frames), _as_indexer(residues), :]

    # Materialize only the sliced region (numpy)
    arr = np.asarray(Zs)  # shape: (F, Lsel, A)

    axis_names = _canon_order_single()  # ["frame","L","A"]

    # WT collapse on A (special)
    if collapse_spec["A"] == "wt":
        if seq is None or aa_order is None:
            raise ValueError("WT collapse requires seq and aa_order.")
        # Map residues selection to correct WT indices
        # residues can be slice or list; turn into explicit indices
        if residues is None:
            res_idx = np.arange(arr.shape[1])
            wt_full = _wt_indices(seq, aa_order)
            wt = wt_full
        else:
            if isinstance(residues, slice):
                res_idx = np.arange(len(seq))[residues]
            else:
                res_idx = np.asarray(residues, dtype=np.int64)
            wt_full = _wt_indices(seq, aa_order)
            wt = wt_full[res_idx]

        # Take along last axis A using per-residue wt (broadcast over frames)
        # arr: (F, Lsel, A), wt: (Lsel,)
        arr = np.take_along_axis(arr, wt[None, :, None].astype(np.int64), axis=2).squeeze(2)
        # Now shape: (F, Lsel)
        axis_names = ["frame", "L"]

    elif collapse_spec["A"] != "keep":
        arr = _apply_reduce(arr, axis=2, mode=collapse_spec["A"])
        # shape: (F, Lsel)
        axis_names = ["frame", "L"]

    # Collapse L if requested (note axis index depends on whether A collapsed)
    if "L" in axis_names and collapse_spec["L"] != "keep":
        ax = axis_names.index("L")
        arr = _apply_reduce(arr, axis=ax, mode=collapse_spec["L"])
        axis_names.pop(ax)  # remove L

    # Collapse frame if requested (you said usually keep trajectory, but support it)
    if "frame" in axis_names and collapse_spec["frame"] != "keep":
        ax = axis_names.index("frame")
        arr = _apply_reduce(arr, axis=ax, mode=collapse_spec["frame"])
        axis_names.pop(ax)

    return arr

def _collapse_pairwise(
    Z: "zarr.Array",
    collapse_spec: Dict[str, str],
    *,
    frames=None,
    residues1=None,
    residues2=None,
    seq=None,
    aa_order=None,
) -> np.ndarray:
    Zs = Z[
        _as_indexer(frames),
        _as_indexer(residues1),
        _as_indexer(residues2),
        :,
        :,
    ]
    arr = np.asarray(Zs)  # shape: (F, L1sel, L2sel, A1, A2)
    axis_names = _canon_order_pairwise()

    # Prepare selected residue indices for WT lookup
    need_wt = (collapse_spec["A1"] == "wt") or (collapse_spec["A2"] == "wt")
    if need_wt:
        if seq is None or aa_order is None:
            raise ValueError("WT collapse requires seq and aa_order.")
        wt_full = _wt_indices(seq, aa_order)

        def _sel_to_indices(sel, L):
            if sel is None:
                return np.arange(L, dtype=np.int64)
            if isinstance(sel, slice):
                return np.arange(L, dtype=np.int64)[sel]
            return np.asarray(sel, dtype=np.int64)

        L = len(seq)
        idx1 = _sel_to_indices(residues1, L)
        idx2 = _sel_to_indices(residues2, L)
        wt1 = wt_full[idx1]  # (L1sel,)
        wt2 = wt_full[idx2]  # (L2sel,)

    # --- WT collapse on A1/A2 (indexing, not reduction) ---
    # Do A2 first or A1 first — either is fine; just keep axis_names updated.

    if collapse_spec["A2"] == "wt":
        # arr: (F, L1, L2, A1, A2), wt2: (L2,)
        # Build indexer of shape (1,1,L2,1,1) to take along axis A2(=4)
        take_idx = wt2[None, None, :, None, None].astype(np.int64)
        arr = np.take_along_axis(arr, take_idx, axis=4).squeeze(4)
        # Now: (F, L1, L2, A1)
        axis_names.remove("A2")

    elif collapse_spec["A2"] != "keep":
        arr = _apply_reduce(arr, axis=axis_names.index("A2"), mode=collapse_spec["A2"])
        axis_names.remove("A2")
        # shape now (F, L1, L2, A1) or (F,L1,L2,A1) depending (same)

    if collapse_spec["A1"] == "wt":
        # After possible A2 removal, A1 axis is the last axis if A2 collapsed;
        # but use axis_names to find it.
        ax = axis_names.index("A1")
        # We need wt1 shape compatible: (1, L1, 1, 1) if arr is (F,L1,L2,A1)
        # Build indexer that matches arr ndim.
        # Current axis order should be subset of ["frame","L1","L2","A1"].
        # We'll create an index array with singleton dims everywhere except L1.
        shape = [1] * arr.ndim
        shape[axis_names.index("L1")] = wt1.shape[0]
        # put wt1 along L1 axis, then add singleton for A1 selection
        # We want take along axis ax, so indexer must broadcast to arr shape except ax dim.
        # A simpler way: expand wt1 to (1,L1,1,1) and take along A1 axis.
        # Ensure it has same ndim as arr:
        wt1_exp = wt1.astype(np.int64)[None, :, None, None]
        # But if L2 was collapsed, arr might be (F,L1,A1) → need (1,L1,1)
        while wt1_exp.ndim < arr.ndim:
            wt1_exp = wt1_exp[..., None]
        # Now take along axis ax:
        arr = np.take_along_axis(arr, wt1_exp, axis=ax).squeeze(ax)
        axis_names.remove("A1")

    elif collapse_spec["A1"] != "keep":
        ax = axis_names.index("A1")
        arr = _apply_reduce(arr, axis=ax, mode=collapse_spec["A1"])
        axis_names.remove("A1")

    # --- Reduce residues axes (L1/L2) ---
    if "L1" in axis_names and collapse_spec["L1"] != "keep":
        ax = axis_names.index("L1")
        arr = _apply_reduce(arr, axis=ax, mode=collapse_spec["L1"])
        axis_names.remove("L1")

    if "L2" in axis_names and collapse_spec["L2"] != "keep":
        ax = axis_names.index("L2")
        arr = _apply_reduce(arr, axis=ax, mode=collapse_spec["L2"])
        axis_names.remove("L2")

    # --- Reduce frame axis ---
    if "frame" in axis_names and collapse_spec["frame"] != "keep":
        ax = axis_names.index("frame")
        arr = _apply_reduce(arr, axis=ax, mode=collapse_spec["frame"])
        axis_names.remove("frame")

    return arr

def _resolve_seq_and_order_for_wt(
    g: "zarr.Group",
    seq: Optional[str],
    aa_order: Optional[str],
) -> tuple[str, str]:
    """
    Resolve WT sequence and amino-acid order for WT-based collapse.

    Precedence:
      seq      : g.attrs["sequence"] -> input arg seq -> error
      aa_order : g.attrs["aa_order"] -> input arg aa_order -> default constant
    """
    # --- sequence ---
    z_seq = g.attrs.get("sequence", None)
    z_seq = str(z_seq).strip() if z_seq is not None else ""

    if z_seq:
        seq_final = z_seq
    elif seq is not None and str(seq).strip():
        seq_final = str(seq).strip()
    else:
        raise ValueError(
            "WT collapse requested (collapse mode 'wt'), but no sequence was provided.\n"
            "Provide `seq=...` or store it in the zarr attrs as root.attrs['sequence']."
        )

    # --- aa_order ---
    z_order = g.attrs.get("aa_order", None)
    z_order = str(z_order).strip() if z_order is not None else ""

    if z_order:
        order_final = z_order
    elif aa_order is not None and str(aa_order).strip():
        order_final = str(aa_order).strip()
    else:
        order_final = AA_DEFAULT_ORDER

    return seq_final, order_final

def collapse_frustration(
    zarr_path: str,
    mode: FrustMode,
    *,
    # selection
    frames: Optional[slice | list[int]] = None,
    residues: Optional[slice | list[int]] = None,        # single
    residues1: Optional[slice | list[int]] = None,       # pairwise
    residues2: Optional[slice | list[int]] = None,       # pairwise

    # main collapse spec (advanced users)
    collapse: Optional[Dict[str, CollapseMode]] = None,

    # convenience shortcuts (basic users)
    collapse_residues: Optional[CollapseMode] = None,
    collapse_amino_acids: Optional[CollapseMode] = None,

    # WT handling
    seq: Optional[str] = None,
    aa_order: Optional[str] = None,

    # output control
    return_kind: Literal["numpy", "npz", "zarr"] = "numpy",
) -> Any:
    """
    Collapse dimensions of a frustration tensor stored in Zarr.

    collapse keys for single:
        "frame", "L", "A"

    collapse keys for pairwise:
        "frame", "L1", "L2", "A1", "A2"

    collapse values:
        "keep", "mean", "sum", "max", "min", "wt"
    """

    collapse_spec = _normalize_collapse(mode, collapse, collapse_residues, collapse_amino_acids)

    g = zarr.open_group(zarr_path, mode="r")
    Z = g["Z"]

    collapse_spec = _normalize_collapse(mode, collapse, collapse_residues, collapse_amino_acids)

    if "wt" in collapse_spec.values():

        seq, aa_order = _resolve_seq_and_order_for_wt(g, seq, aa_order)
        _wt_indices(seq, aa_order)  # validate seq and aa_order against each other

    if mode == "single":
        out = _collapse_single(
            Z, collapse_spec,
            frames=frames, residues=residues,
            seq=seq, aa_order=aa_order,
        )
    else:
        out = _collapse_pairwise(
            Z, collapse_spec,
            frames=frames, residues1=residues1, residues2=residues2,
            seq=seq, aa_order=aa_order,
        )

    return out