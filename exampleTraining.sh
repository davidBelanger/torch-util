d=./proc/

lr=0.1
labelDim=`cat $d/domain.labelDomainSize.txt`
vocabSize=`cat proc/domain.domainSizes.txt  | grep '^tokenString' | cut -f2`

embeddingDim=25
options="-trainList $d/train.list -testList $d/dev.list -minibatch 32 -cuda 0 -labelDim $labelDim -vocabSize $vocabSize -learningRate $lr"

th ConvExample3.lua $options

