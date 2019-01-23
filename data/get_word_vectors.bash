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

curl -LO $fasttextpath
unzip crawl-300d-2M.vec.zip 
rm crawl-300d-2M.vec.zip

curl -LO $GoogleNewspath
gunzip GoogleNews-vectors-negative300.bin.gz
python -c "from gensim.models import word2vec; model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True); model.save("GoogleNews-vectors-negative300.txt")"

cd ../auxiliary_data
curl -LO $SIF
cd ..
