d=./proc/

lr=0.1
labelDim=`cat $d/domain.labelDomainSize.txt`
vocabSize=`cat proc/domain.domainSizes.txt  | grep '^tokenString' | cut -f2`

tmp=/tmp/domainSizes

echo tokenString:25 > $tmp
echo isCap:5 >> $tmp
echo isNumeric:3 >> $tmp

cat $tmp | tr ':'  '\t' > $tmp.2

echo python mergeMaps.py $d/domain.domainSizes.txt $tmp.2
python mergeMaps.py $d/domain.domainSizes.txt $tmp.2 > $tmp.merge

cat $tmp.merge
options="-trainList $d/train.list -testList $d/dev.list -minibatch 32 -cuda 0 -labelDim $labelDim -vocabSize $vocabSize -learningRate $lr -featureEmbeddings 1 -featureEmbeddingSpec $tmp.merge"

th ConvExample3.lua $options

