# ICML-2019 Supporting Code for Submission #960

This repo consits of a slight set of modifications for [SentEval](https://github.com/facebookresearch/SentEval) to reproduce results in anonymous submission #960 @ICML2019. 

**NOTE:** This repo is a **fork** of [SentEval](https://github.com/facebookresearch/SentEval). Any commits prior to 2019 are not associated to submission #960 @ICML2019. 

## Dependencies

This code is written in python. The dependencies are:

* Python 2 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0
* [Autograd](https://github.com/HIPS/autograd/)


## Download datasets
To get all the transfer tasks datasets, run (in data/) using Bash >= 4.0:
```bash
./get_transfer_data_ptb.bash
```
This will automatically download and preprocess the datasets, and store them in data/senteval_data (warning: for MacOS users, you may have to use p7zip instead of unzip). Note: we provide PTB or MOSES tokenization.

WARNING: Extracting the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) MSI file requires the "[cabextract](https://www.cabextract.org.uk/#install)" command line (i.e *apt-get/yum install cabextract*).

This will also download **glove.840B.300d.txt** and **enwiki_vocab_min200.txt** (The SIF frequencies from Arora et al. 2016).

To download the other word vectors please go to [GoogleNews-word2vec](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download) and [FastText](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip), then convert binary files into the same text file format as **glove.840B.300d.txt** and place in **/data/word_vectors**. We could not upload them to GitHub since they are above the allowed disk-quota.

### Reproduce Results for Submission #960 

In order to reproduce results please run:

```bash
cd examples
python arora.py # To reproduce Arora et al. (2016)'s SIF+PCA results
python gaussian.py # To reproduce our Gaussian-AIC/TIC results
python vmf.py  # To reproduce our vMF-AIC/TIC results
```

This will reproduce the results for **glove.840B.300d.txt** and potentially crash afterwards if you have not downloaded the other word vectors. 

### Similarity code
The entire codebase connected to the similarity metrics described in the paper is encapsulated in the *similarity* folder. This is where the core contributions of our work are.
