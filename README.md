# ICML-2019 supporting code submission for paper #960

This repo consits of a slight set of modifications for [SentEval](https://github.com/facebookresearch/SentEval) to reproduce results in anonymous submission #960 @ICML2019. 

## Dependencies

This code is written in python. The dependencies are:

* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0
* [Autograd](https://github.com/HIPS/autograd/)


Note: COCO comes with ResNet-101 2048d image embeddings. [More details on the tasks.](https://arxiv.org/pdf/1705.02364.pdf)

## Download datasets
To get all the transfer tasks datasets, run (in data/):
```bash
./get_transfer_data_ptb.bash
```
This will automatically download and preprocess the datasets, and store them in data/senteval_data (warning: for MacOS users, you may have to use p7zip instead of unzip). Note: we provide PTB or MOSES tokenization.

WARNING: Extracting the [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) MSI file requires the "[cabextract](https://www.cabextract.org.uk/#install)" command line (i.e *apt-get/yum install cabextract*).

### Reproduce Paper Results

In order to reproduce results please run:

```bash
cd examples
python arora.py # To reproduce SIF's arora et al results
python gaussian_aic.py # To reproduce our Gaussian-AIC results
python vmf.py  # To reproduce our vMF-TIC results
```

This will reproduce the results for **glove.840B.300d.txt** and potentially crash afterwards if you have not downloaded the other word vectors. 
