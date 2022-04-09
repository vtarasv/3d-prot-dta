<h1 align="center">
<p> Drug-Target Affinity Prediction Using Graph Neural Networks and AlphaFold-Based Protein Structures</h1>

<p align="center"><img src="img/general_model.png" alt="logo" width="700px" /></p>

 ---

# Requirements
- python 3.7+ <br />
# Usage
1. Clone the repository <br />
`git clone https://github.com/vtarasv/3d-prot-dta.git` <br />
2. Create and activate new python virtual environment <br />
3. Install requited packages <br />
`pip install -r 3d-prot-dta/requirements.txt` <br />
4. Run experiments <br />
`python 3d-prot-dta/test.py -b davis` for the Davis dataset <br />
`python 3d-prot-dta/test.py -b kiba` for the KIBA dataset <br />
The results will be saved in the `3d-prot-dta/results/` <br />
# Data
See the corresponding [README](data/README.md)
# Notes
- It is recommended to use GPU to speed up the experiments (machines with 1 GPU perform 20 times faster on average than machines with 4 CPUs)
- The code is tested on Ubuntu and Microsoft Windows operating systems
