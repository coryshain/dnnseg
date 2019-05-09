# DNNSeg

DNNSeg implements deep neural sequence models for unsupervised speech processing and for testing hypotheses
about human language acquisition from raw speech. DNNSeg is an elaboration on the model described in
Elsner & Shain (2017; see [implementation](https://github.com/melsner/neural-segmentation)), and a constrained variant
of the DNNSeg model has been used to study the acquisition of phonological categories and features from speech 
(Shain & Elsner, 2019). In its full form, DNNSeg infers hierarchically organized segment boundaries and category
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

 - `naacl19`

Thus, to reproduce results from NAACL19 (Shain & Elsner, 2019), for example, run `git checkout naacl19` from the repository root, and follow instructions in the `README` file.

Published results depend on both (1) datasets and (2) models as defined in experiment-specific configuration files.
We do not distribute data with this repository.

## References
* Baddeley, Alan; Gathercole, Susan; and Papagno, Costanza (1998). The phonological loop as a language learning device. _Psychological Review_.
* Chung, Junyoung; Ahn, Sungjin; and Bengio, Yoshua. Hierarchical multiscale recurrent neural networks. _ICLR17_.
* Elsner, Micha and Shain, Cory (2017). Speech segmentation with a neural encoder model of working memory. _EMNLP17_.
* Shain, Cory and Elsner, Micha (2019). Measuring the perceptual availability of phonological features during language
  acquisition using unsupervised binary stochastic autoencoders. _NAACL19_.
* Singer, Yosef; Teramoto, Yayoi; Willmore, Ben D B; Schnupp, Jan W H; King, Andrew J; Harper, Nicol S. Sensory cortex is optimized for prediction of future input. _eLife_.