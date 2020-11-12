# DNNSeg: Shain & Elsner '20 Reproduction Branch

This is the reproduction branch for Shain & Elsner (2020). For the current version of the model, run:

    git checkout master

Shain & Elsner (2020) used DNNSeg to test existing hypotheses about the roles of memory and
prediction in child phoneme acquisition.

## Installation

Install DNNSeg by cloning this repository. In addition, DNNSeg requires the Python libraries tensorflow, pandas,
scpiy, scikit-learn, librosa, and seaborn, along with their dependencies. These can be installed using pip or conda.
Furthermore, for models using cochleagram-based acoustic representations, you will need to install the pycochleagram
library by running the following in the repository root:

    git clone https://github.com/mcdermottLab/pycochleagram.git;
    cd pycochleagram;
    python setup.py install
    

## Data Setup

Running DNNSeg on the Zerospeech 2015 challenge data requires four external resources:

  - The [Zerospeech metadata](https://github.com/bootphon/Zerospeech2015)
  - The [Zerospeech track 2 repository](https://github.com/bootphon/tde)
  - The [Buckeye Speech Corpus](https://buckeyecorpus.osu.edu/)
  - The Xitsonga portion of the [NCHLT corpus](https://repo.sadilar.org/handle/20.500.12185/277)
  
Once these have been acquired and downloaded, they should be preprocessed by running the following from the 
DNNSeg repository root:

    python -m dnnseg.datasets.zerospeech.build <PATH-TO-ZS-METADATA> <PATH-TO-ZS-TRACK2> -b <PATH-TO-BSC> -x <PATH-TO-NCHLT> -o <PATH-TO-OUTPUT-DIR>


## Fitting Models

Model data and hyperparameters are defined in `*.ini` config files. For an example config file, see `dnnseg_model.ini`
in the repository root. For a full description of all settings that can be controlled with the config file,
see the DNNSeg initialization params by running:

    python3 -m dnnseg.bin.help
    
Once you have defined an `*.ini` config file, fit the model by running the following from the repository root:

    python3 -m dnnseg.bin.train <PATH>.ini

## Reproducing Shain & Elsner (2020)

If necessary, install [anaconda](https://www.anaconda.com/), then create and activate a conda environment
for DNNSeg:

    conda env create -f conda_conll20.yml
    conda activate dnnseg

Then run:

    make

This generates config files for several DNNSeg models (`*.ini`) in a `conll20` directory.
Each can be run individually as

    python -m dnnseg.bin.train conll20/<MODEL>.ini

To run all models concurrently on a Portable Batch System style cluster, first run:

    make pbs

which will generate corresponding PBS script for each model, then run:

    ./qsub.sh conll20/*pbs

which will submit all jobs concurrently.


## References
* Shain, Cory and Elsner, Micha (2020). Acquiring language from speech by learning to remember and predict. _CoNLL20_.
