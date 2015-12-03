#!/bin/sh
set -e

##specification for the output data paths
outDir=proc/
name=example #use this to give some informative name to the processed data files

##specifications about the input data
trainFile=pos/trn.pos.proc
devFile=pos/dev.pos.proc
testFile=pos/tst.pos.proc

tokLabels=1 #whether the input has labels at the token level (alternative: at the sentence level)
allFiles="$trainFile:train $devFile:dev $testFile:test" #if there are more files (eg a second dev set) just specify it here


##specification about features
tokFeats=0 #whether to use features for each token (alternative: just token string, ie word type, is the only feature)
featureTemplates=tokenString,isCap #if using token features, this is a list of the names of the templates to use (assuming that each of these is implemented in $makeFeatures)

#this is an example of all the implemented options for features. Here, you can use any width d for the final features: Prefix-d  and Suffix-d
#featureTemplates=tokenString,isCap,isNumeric,Prefix-3,Suffix-3 

##parameters to choose
featureCountThreshold=5
lengthRounding=5 #this pads such that every token and label sequence has a length that is a multiple of <lengthRounding> (only used on train data)
pad=0 #this puts <pad> dummy tokens on each side (important for CNNs). 

#verbosity level
verbose=1

dataPaths="-trainFile $trainFile -devFile $devFile -testFile $testFile -name $name -outDir $outDir"
dataOptions="-tokLabels $tokLabels -tokFeats $tokFeats -featureTemplates $featureTemplates -featureCountThreshold $featureCountThreshold -lengthRounding $lengthRounding -pad $pad"

#command to be run
cmd="python dataProcessing.py  $dataPaths $dataOptions -verbose $verbose"

echo executing:
echo $cmd
$cmd


