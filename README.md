# DNNSeg

DNNSeg is a repository for unsupervised speech segmentation and classification using deep neural frameworks.
You are currently in the `NAACL19` branch, which is exclusively used for reproducing results from Shain & Elsner (2019).
Do not use this branch to train new models on your own data.
Intead, first run the following from the repository root:

`git checkout -b master`

To reproduce Shain & Elsner (2019), you will first need the data from the [Zerospeech 2015 challenge](https://github.com/bootphon/Zerospeech2015).
Once the data are in hand, you will need to run the following preprocessing script from the repository root:

`python -m dnnseg.datasets.zerospeech.build -b <PATH-TO-BUCKEYE-ROOT> -x <PATH-TO-NCHLT-ROOT>`

Once the data have been preprocessed, models can be trained by running:

`python -m dnnseg.bin.train <PATH-TO-INI-FILE>`

The eight models of Shain & Elsner (2019) are defined in the following files at the repository root:

  - `english_zerospeech_classify.ini`
  - `english_zerospeech_classify_nospeaker.ini`
  - `english_zerospeech_classify_nobsn.ini`
  - `english_zerospeech_classify_nospeaker_nobsn.ini`
  - `xitsonga_zerospeech_classify.ini`
  - `xitsonga_zerospeech_classify_nospeaker.ini`
  - `xitsonga_zerospeech_classify_nobsn.ini`
  - `xitsonga_zerospeech_classify_nospeaker_nobsn.ini`

Questions and feedback can be directed to Cory Shain ([shain.3@osu.edu](shain.3@osu.edu)).

## References

* Shain, Cory and Elsner, Micha (2019). Measuring the perceptual availability of phonological features during language
  acquisition using unsupervised binary stochastic autoencoders. _NAACL19_.
