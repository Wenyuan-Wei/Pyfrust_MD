from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Iterable, Tuple

# Standard 3-letter to 1-letter map (+ common variants)
_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "HID": "H", "HIE": "H", "HIP": "H",  # AMBER
    "HSD": "H", "HSE": "H", "HSP": "H",  # CHARMM
    "CYX": "C", "CYM": "C", "CYP": "C",  # seen in some workflows
    # Common/modified residues you likely want to treat as standard:
    "MSE": "M",  # selenomethionine
    "SEC": "U",  # selenocysteine (optional; change to "C" if you prefer)
    "PYL": "O",  # pyrrolysine (optional; rarely in PDB)
}

def _iter_pdb_ca_residues(
    pdb_path: str | Path,
    chain: Optional[str],
    *,
    use_model: int = 1,
    allow_altloc: Tuple[str, ...] = (" ", "A"),
) -> Iterable[Tuple[str, str, str, str]]:
    """
    Yield (chain_id, resname, resseq, icode) for each *new* residue seen via CA atoms.
    Uses only the requested MODEL (default 1) if MODEL records exist.
    """
    pdb_path = Path(pdb_path)

    in_model = True
    model_idx = 0
    seen = set()  # (chain, resseq, icode) per model

    with pdb_path.open("rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = line[:6]
            if rec.startswith("MODEL"):
                # MODEL        1
                try:
                    model_idx = int(line[10:14].strip())
                except ValueError:
                    model_idx += 1
                in_model = (model_idx == use_model)
                continue
            if rec.startswith("ENDMDL"):
                if in_model:
                    break
                continue

            if not in_model:
                continue

            if rec != "ATOM  ":
                continue

            atom_name = line[12:16]
            if atom_name != " CA ":
                continue

            altloc = line[16:17]
            if altloc not in allow_altloc:
                continue

            chain_id = line[21:22]
            if chain is not None and chain_id != chain:
                continue

            resname = line[17:20].strip().upper()
            resseq = line[22:26].strip()
            icode = line[26:27].strip()  # insertion code (may be "")

            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen.add(key)

            yield chain_id, resname, resseq, icode


def get_sequence_fast(
    pdb_file: str | Path,
    chain: Optional[str] = None,
    *,
    unknown: str = "X",
) -> str:
    """
    Fast sequence extraction with *no major dependencies*.
    - For .pdb/.ent: streams ATOM CA records.
    - For .cif/.mmcif: raises by default (use a parser library or MDAnalysis fallback).
    """
    pdb_file = str(pdb_file)
    suffix = Path(pdb_file).suffix.lower()

    if suffix in {".cif", ".mmcif"}:
        raise ValueError(
            "mmCIF parsing without a dedicated library is not implemented in get_sequence_fast(). "
            "Use get_sequence_mdanalysis() if your MDAnalysis install can read mmCIF, "
            "or use BioPython/gemmi for mmCIF."
        )

    seq_chars: list[str] = []
    for _chain_id, resname, _resseq, _icode in _iter_pdb_ca_residues(pdb_file, chain):
        seq_chars.append(_THREE_TO_ONE.get(resname, unknown))

    return "".join(seq_chars)


def get_sequence_mdanalysis(
    structure_file: str | Path,
    chain: Optional[str] = None,
    *,
    unknown: str = "X",
) -> str:
    """
    Sequence extraction using MDAnalysis (fast; depends on your installed readers).
    Works best for PDB; mmCIF support varies by setup.
    """
    import MDAnalysis as mda

    u = mda.Universe(str(structure_file))

    # Select protein-ish atoms. We take CA per residue to avoid duplicates and keep order.
    sel = "protein and name CA"
    if chain is not None:
        # MDAnalysis uses segid/chainID depending on topology; try chainID first.
        # Many PDBs map chain IDs to 'segid' or 'chainID'. We'll try both robustly.
        try_sel = f"({sel}) and (chainID {chain})"
        ag = u.select_atoms(try_sel)
        if len(ag) == 0:
            ag = u.select_atoms(f"({sel}) and (segid {chain})")
    else:
        ag = u.select_atoms(sel)

    # Build 3-letter residue names in residue order
    seq = []
    for res in ag.residues:
        resname = (res.resname or "").strip().upper()
        seq.append(_THREE_TO_ONE.get(resname, unknown))
    return "".join(seq)


def get_sequence(
    pdb_file: str | Path,
    chain: Optional[str] = None,
    *,
    prefer: str = "fast",  # "fast" or "mdanalysis"
    unknown: str = "X",
) -> str:
    """
    Convenience wrapper:
    - prefer="fast": use streaming PDB parser for .pdb; otherwise try MDAnalysis.
    - prefer="mdanalysis": try MDAnalysis first; fallback to fast PDB parser.
    """
    suffix = Path(str(pdb_file)).suffix.lower()

    if prefer == "mdanalysis":
        try:
            return get_sequence_mdanalysis(pdb_file, chain, unknown=unknown)
        except Exception:
            if suffix not in {".cif", ".mmcif"}:
                return get_sequence_fast(pdb_file, chain, unknown=unknown)
            raise

    # prefer == "fast"
    if suffix not in {".cif", ".mmcif"}:
        return get_sequence_fast(pdb_file, chain, unknown=unknown)

    # mmCIF: try MDAnalysis (if available), otherwise raise
    return get_sequence_mdanalysis(pdb_file, chain, unknown=unknown)