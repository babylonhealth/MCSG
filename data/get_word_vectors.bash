glovepath='http://nlp.stanford.edu/data/glove.840B.300d.zip'
fasttextpath='https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip'
GoogleNewspath='https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz'
SIF='https://github.com/PrincetonML/SIF/raw/master/auxiliary_data/enwiki_vocab_min200.txt'


mkdir word_vectors
mkdir auxiliary_data
cd word_vectors

curl -LO $glovepath
unzip glove.840B.300d.zip 
rm glove.840B.300d.zip


