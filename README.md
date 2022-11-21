<h1 align="center">
<p> Drug-Target Affinity Prediction Using Graph Neural Networks and AlphaFold-Based Protein Structures</h1>

<p align="center"><img src="img/general_model.png" alt="logo" width="700px" /></p>

 ---

# Requirements
- python 3.7+ <br />
# Usage
1. Clone the repository, navigate to the cloned folder <br />
`git clone https://github.com/vtarasv/3d-prot-dta.git` <br />
`cd 3d-prot-dta/` <br />
2. Create and activate a new python virtual environment <br />
3. Install required packages <br />
`pip install wheel` <br />
`pip install -r requirements.txt` <br />
4. Run the experiments <br />
`python test.py` to obtain test datasets results as described in the manuscript <br />
`python test.py --datasets davis` to obtain only Davis test dataset results <br />
`python test.py --datasets kiba` to obtain only KIBA test dataset results <br />
The results will be saved in the `results/` folder <br />
The log will be saved in the `log/` folder <br />
5. You can also launch the tuning process in the same way as described in the manuscript <br />
`python tune.py --study my_study --sampler tpe` <br />
The tuning results will be saved in the local storage `sqlite:///dta_tune.db` (in the same folder) <br />
# Data
See the corresponding [README](data/README.md)
# Notes
- It is recommended to use GPU to speed up the experiments (machines with 1 GPU perform 20 times faster on average than machines with 4 CPUs)
- The training of 5 models (one per cross-validation train dataset) using one NVIDIA Tesla P100 SXM2 GPU takes about 10 and 40 hours for Davis and KIBA datasets respectively 
- The code is tested on Ubuntu operating system