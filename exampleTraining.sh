#!/bin/sh

#input data
d=./proc3/ #this is where the processed data is
tokFeats=1 #use whatever the prepocessing did
tokLabels=1

#training options
lr=0.1 #learning rate. TODO: make a more verbose framework for specifying optimization options on the command line
gpuid=-1 #if >= 0, then do computation on GPU
minibatch=32 #if using gpu, minibatch sizes needs to be a multiple of 32.
##note: if using token features, you can also tweak the per-feature-template embedding sizes below (eg, tokenString:25)

labelDim=`cat $d/domain.labelDomainSize.txt`
vocabSize=`cat $d/domain.domainSizes.txt  | grep '^tokenString' | cut -f2`

#see ModelTraining.lua for documentation of its command line options
options="-trainList $d/train.list -testList $d/dev.list -tokenFeatures $tokFeats -tokenLabels $tokLabels -minibatch $minibatch -gpuid $gpuid -labelDim $labelDim -vocabSize $vocabSize -learningRate $lr"

if [ "$tokFeats" == "0" ]; then
	embeddingDim=25
	featureDim=15
	moreOptions="-embeddingDim $embeddingDim -featureDim $featureDim"
	options="$options $moreOptions"
else
	tmp=/tmp/domainSizes
	echo tokenString:25 > $tmp
	echo isCap:5 >> $tmp
	echo isNumeric:3 >> $tmp
	cat $tmp | tr ':'  '\t' > $tmp.2
	python mergeMaps.py $d/domain.domainSizes.txt $tmp.2 > $tmp.merge

	moreOptions="-featureEmbeddingSpec $tmp.merge"
	options="$options $moreOptions"
fi

th ModelTraining.lua $options

#th ModelTraining.lua $options

