### Dataset Files

- **H2O_Pickle.zip**: Contains the training, validation, and test datasets for the Torsional Diffusion model in the solvent which are preprocessed based on Conf_Solv dataset **H2O**.
- **Hexane_pickle.zip**: Contains the training, validation, and test datasets for the Torsional Diffusion model in the solvent which are preprocessed based on Conf_Solv dataset **Hexane**.

### Molecular Data Files

- **drugs_20steps_hexane_mol1.pkl**, **drugs_20steps_shui_mol1.pkl**: Generated Conformers for the molecule `O=C(O)C1=CC(OP(=O)(O)O)CCC1` by our finetuned Torsional Diffusion model. The conformers with lowest mmff energy in each solvent (H2O and Hexane) were used in our Visualization part.

- **drugs_20steps_shui_mol2.pkl**, **drugs_20steps_hexane_mol2.pkl**: Generated Conformers for the molecule `[H]OC(=O)C1=C([H])[C@@]([H])(OP(=O)(O[H])O[H])C([H])([H])C([H])([H])C1([H])[H]` by our finetuned Torsional Diffusion model. The conformers with lowest mmff energy in each solvent (H2O and Hexane) were used in our Visualization part. 
