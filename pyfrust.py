import frustratometer
import pathlib
import numpy as np
import contextlib
import io
import os 
from typing import Optional, Tuple

# Global variable for amino acid indexing
# This is consistent with the indexing used in the frustratometer package, where the first character is a placeholder for gaps or unknown residues.
_AA = "-ACDEFGHIKLMNPQRSTVWY"

def single_frust(pdb:str, chain:Optional[str]=None, 
                 k_electrostatics:float=17.3636, 
                 min_sequence_separation_contact:int=2, 
                 validate:bool=False)->Tuple[np.ndarray, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        # Read the structure from the PDB file
        pdb=pathlib.Path(pdb)
        structure=frustratometer.Structure(pdb_file=pdb, chain=chain)
        os.remove(f"{pdb.stem}_cleaned.pdb")  # Clean up the intermediate cleaned PDB file
        ## Single residue frustration with electrostatics
        ## Immediate neighbors are ignored -- likely to suppress secondary structure effects 
        model_singleresidue = frustratometer.AWSEM(structure, min_sequence_separation_contact=min_sequence_separation_contact, 
                                                   k_electrostatics=k_electrostatics)

        # Calculate AWSEM energy change with respect to wildtype 
        DE=model_singleresidue.decoy_fluctuation(kind="singleresidue")
    
    # Obtain amino acid frequencies from the model
    aa_freq=model_singleresidue.aa_freq
    # Normalize to get probabilities
    reweighted_aa_freq=aa_freq / aa_freq.sum()  
    # Get wildtype amino acid index
    wt_indx = np.fromiter(( _AA.index(i) for i in structure.sequence), 
                          dtype=int, count=len(structure.sequence))
    
    mean = DE @ reweighted_aa_freq
    std = np.sqrt(((DE - mean[:, None])**2 @ reweighted_aa_freq))
    std = np.where(std == 0, np.nan, std)  # Avoid division by zero

    Z = (mean[:, None] - DE) / std[:, None]

    if validate:
        library_z=model_singleresidue.frustration(kind="singleresidue")
        assert np.allclose(Z[np.arange(len(structure.sequence)), wt_indx], library_z, equal_nan=True), "Calculated Z-scores do not match library values."
    
    # Return the Z-scores for all residues and the Z-score for the wildtype amino acid 
    return Z, Z[np.arange(len(structure.sequence)), wt_indx]

def pairwise_frust():pass 


# Placeholder for comformational frustration calculation
# For now it is not developed 
def pairwise_comf_frust():pass 