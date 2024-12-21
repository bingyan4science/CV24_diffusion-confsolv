import os
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch
from ase.atoms import Atoms

from conf_solv.trainer import LitConfSolvModule
from conf_solv.dataloaders.loader import create_pairdata
from conf_solv.dataloaders.features import MolGraph
from conf_solv.model.model import ConfSolv
from joblib import Parallel, delayed

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rdkit_mol_to_ase_atoms(mol):
    """Convert RDKit molecule to ASE Atoms object"""
    positions = mol.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)

def flatten(xss):
    return [x for xs in xss for x in xs]

def load_lightning_model(trained_model_dir, i):
    #import ipdb; ipdb.set_trace()
    models = [LitConfSolvModule.load_from_checkpoint(os.path.join(trained_model_dir, f'ensemble_{i}','best_model.ckpt'))]
    return models

def load_lightning_model_parallel(trained_model_dir, ensemble_nos):
    models = Parallel(n_jobs=len(ensemble_nos))([delayed(load_lightning_model)(trained_model_dir,i) for i in ensemble_nos])
    models = flatten(models)
    return models

class ConfsolvSinglePrediction:
    def __init__(self, num_cores: int = -1) -> None:
        self.models: Optional[List[ConfSolv]] = None
        self.model_parameters: Optional[Any] = None
        self.trained_model_dir = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if num_cores != -1:
            torch.set_num_threads(num_cores)
        
        #print(f'Device being used is {self.device}')
        #print(f'Number of threads used by torch is {torch.get_num_threads()}')
        
    def load_models(self, trained_model_dir: str = 'confsolv_models/nonionic_solvents_scaffold') -> None:
        self.trained_model_dir = Path(trained_model_dir)
        ensemble_nos = len([x for x in self.trained_model_dir.iterdir() if x.is_dir()])
        #print(f"Loading model {self.trained_model_dir.name} with {ensemble_nos} ensembles...")
        self.models = load_lightning_model_parallel(trained_model_dir, range(ensemble_nos))

    def predict(self, rdkit_mol, solvent_smiles: str) -> Tuple[float, float]:
        """
        Predict solvation for a single RDKit molecule in a specific solvent
        
        Parameters:
        -----------
        rdkit_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object with 3D coordinates
        solvent_smiles : str
            SMILES string of the solvent
            
        Returns:
        --------
        prediction : float
            Solvation prediction
        uncertainty : float
            Standard deviation of the prediction
        """
        #print("Running prediction...")
        #import ipdb; ipdb.set_trace()
        
        # Convert RDKit mol to ASE Atoms
        ase_mol = rdkit_mol_to_ase_atoms(rdkit_mol)
        
        # Make prediction
        solvent_molgraph = MolGraph(solvent_smiles)
        data = create_pairdata(solvent_molgraph, [ase_mol], 1)
        data.solute_confs_batch = torch.zeros([ase_mol.get_global_number_of_atoms()], dtype=torch.int64)
        batch_data = Batch.from_data_list([data], follow_batch=['x_solvent', 'x_solute'])
        
        if self.device == 'cuda':
            batch_data.cuda()
            for model in self.models:
                model.cuda()
                
        with torch.no_grad():
            out = torch.stack([model(batch_data, 1) for model in self.models])
        
        out = out.cpu()
        #out_scaled = out - out.min(dim=1, keepdim=True).values
        out_scaled = out
        #std = out_scaled.std(dim=0)
        std = 0.0
        pred = out_scaled.mean(dim=0)
        #pred = pred - pred.min()
        
        return pred.item(), std

def predict_single(rdkit_mol, solvent_smiles, model_path='../sample_trained_models/', num_cores=-1):
    """
    Convenience function to predict solvation for a single molecule in a specific solvent
    
    Parameters:
    -----------
    rdkit_mol : rdkit.Chem.rdchem.Mol
        RDKit molecule with 3D coordinates
    solvent_smiles : str
        SMILES string of the solvent
    model_path : str
        Path to model directory
    num_cores : int
        Number of CPU cores to use (-1 for all)
        
    Returns:
    --------
    prediction : float
        Solvation prediction
    uncertainty : float
        Prediction uncertainty
    """
    seed_everything(seed=10608)
    
    predictor = ConfsolvSinglePrediction(num_cores=num_cores)
    
    #print("Loading models...")
    predictor.load_models(model_path)
    
    #print("Making prediction...")
    return predictor.predict(rdkit_mol, solvent_smiles)

if __name__ == "__main__":
    # Example usage:
    """
    import rdkit.Chem as Chem
    #import ipdb; ipdb.set_trace()
    
    # Load your molecule
    mol_dir = '/scratch/by2192/torsional-diffusion'
    for i in range(11):
        mol_file = os.path.join(mol_dir, f'molecule_{i}.mol')
        with open(mol_file, 'r') as f:
            mol_block = f.read()
        mol = Chem.MolFromMolBlock(mol_block)
    
        # Define your solvent (using SMILES)
        solvent_smiles = "O"  # ethanol
        
        # Make prediction
        prediction, uncertainty = predict_single(
            mol,
            solvent_smiles,
            model_path="../sample_trained_models/",
            num_cores=-1
        )
        
        print(f"Prediction: {prediction:.4f} ± {uncertainty:.4f}")
    """