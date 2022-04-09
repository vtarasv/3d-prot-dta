# Ligands data
- `davis/smiles_to_graph.pkl` | `kiba/smiles_to_graph.pkl` - dictionary with ligand SMILES strings as keys and ligand graphs (nodes features, edges, edges features) as values
- `davis/smiles_iso_to_ecfp.pkl` | `kiba/smiles_iso_to_ecfp.pkl` - dictionary with ligand isomorphic SMILES strings as keys and Morgan fingerprints (2048-bit vector) as values
# Proteins data
- `davis/protid_to_graph.pkl` | `kiba/protid_to_graph.pkl` - dictionary with unique protein identifiers as keys and protein graphs (nodes features, edges, edges features) as values
- `davis/protid_to_meta.pkl` | `kiba/protid_to_meta.pkl` - dictionary with unique protein identifiers as keys and protein metadata as values
# Other data
- `davis/full.csv` | `kiba/full.csv` - tables with the protein-ligand pairs and corresponding affinity values (labels)
- `data/davis/folds` | `data/kiba/folds` - train and test folds of the datasets used for cross-validation and testing
- `prot_3d_for_Davis.tar.gz` | `prot_3d_for_KIBA.tar.gz` - filtered PDB structures generated with AlphaFold and used to create protein graphs
