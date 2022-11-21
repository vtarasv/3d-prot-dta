# Ligands data
- `davis/ligand_to_graph.pkl` | `kiba/ligand_to_graph.pkl` - python dictionary with ligand SMILES strings as keys and ligand graphs (nodes features, edges, edges features) as values
- `davis/ligand_to_ecfp.pkl` | `kiba/ligand_to_ecfp.pkl` - python dictionary with ligand isomorphic SMILES strings as keys and Morgan fingerprints (2048-bit vector) as values

# Proteins data
- `davis/protein_to_graph.pkl` | `kiba/protein_to_graph.pkl` - python dictionary with unique protein identifiers as keys and protein graphs (nodes features, edges, edges features) as values

# Other data
- `davis/full.csv` | `kiba/full.csv` - tables with the protein-ligand pairs and corresponding affinity values (labels)
- `davis/folds/` | `kiba/folds/` - train and test folds of the datasets used for cross-validation and testing
- `prot_3d_for_Davis.tar.gz` | `prot_3d_for_KIBA.tar.gz` - filtered PDB structures generated with AlphaFold and used to create protein graphs
