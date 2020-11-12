# DNNSeg

DNNSeg implements deep neural sequence models for unsupervised speech processing and for testing hypotheses
about human language acquisition from raw speech. DNNSeg is an elaboration on the model described in
Elsner & Shain (2017; see [implementation](https://github.com/melsner/neural-segmentation)), and variants
have been used to study the acquisition of phonological categories and features from speech 
(Shain & Elsner, 2019; Shain & Elsner, 2020). In its full form, DNNSeg infers hierarchically organized segment boundaries and category
labels through end-to-end optimization of cognitively-inspired proxy objectives for compression (Baddeley et al., 1998)
and predictive coding (Singer et al., 2018), using a special type of segmental recurrent unit (Chung et al., 2017).
DNNSeg is thus based on the hypothesis that linguistic representations (e.g. phonemes, words, and possibly constituents)
make the speech signal both easier to remember and easier to predict than non-linguistic ones, and it exploits this
signal to extract linguistic generalizations from speech without supervision.

## Reproducing published results

This repository is under active development, and reproducibility of previously published results is not guaranteed from the master branch.
For this reason, repository states associated with previous results are saved in Git branches.
To reproduce those results, checkout the relevant branch and follow the instructions in the `README`.
Current reproduction branches are:

 - `NAACL19`
 - `CoNLL20`

Thus, to reproduce results from CoNLL20 (Shain & Elsner, 2020), for example, run `git checkout CoNLL20` from the repository root, and follow instructions in the `README` file.

Published results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
We do not distribute data with this repository.

## Installation

Install DNNSeg by cloning this repository. In addition, DNNSeg requires the Python libraries tensorflow, pandas,
scpiy, scikit-learn, librosa, and seaborn, along with their dependencies. These can be installed using pip or conda.
Furthermore, for models using cochleagram-based acoustic representations, you will need to install the pycochleagram
library by running the following in the repository root:

    git clone https://github.com/mcdermottLab/pycochleagram.git;
    cd pycochleagram;
    python setup.py install

Use [conda](https://www.anaconda.com/) to set up the software environment, like so:

    conda env create -f conda_dnnseg.yml
    conda activate dnnseg
 

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


## References
* Baddeley, Alan; Gathercole, Susan; and Papagno, Costanza (1998). The phonological loop as a language learning device. _Psychological Review_.
* Chung, Junyoung; Ahn, Sungjin; and Bengio, Yoshua. Hierarchical multiscale recurrent neural networks. _ICLR17_.
* Elsner, Micha and Shain, Cory (2017). Speech segmentation with a neural encoder model of working memory. _EMNLP17_.
* Shain, Cory and Elsner, Micha (2019). Measuring the perceptual availability of phonological features during language
  acquisition using unsupervised binary stochastic autoencoders. _NAACL19_.
* Shain, Cory and Elsner, Micha (2020). Acquiring language from speech by learning to remember and predict. _CoNLL20_.
* Singer, Yosef; Teramoto, Yayoi; Willmore, Ben D B; Schnupp, Jan W H; King, Andrew J; Harper, Nicol S. Sensory cortex is optimized for prediction of future input. _eLife_.
