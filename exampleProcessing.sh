#!/bin/sh

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
pad=1 #this puts <pad> dummy tokens on each side (important for CNNs). 


#script paths
makeFeatures="python featureExtraction.py"
addOne="-addOne 1" #use this if preprocessing is 0-indexed. set this to the empty string if your preprocessing is 1-indexed. (featureExtraction.py is 0-indexed)
splitByLength="python splitByLength.py"
int2torch="th int2torch.lua"
domainName=$outDir/domain

featureSpec="-tokenFeatures $tokFeats  -tokenLabels $tokLabels -featureTemplates $featureTemplates"

#---------------Below this are things that you shouldn't need to change-------------------#

##first, establish string->int mappings for the feature templates
makeDomain=1
output="/dev/null"
echo $makeFeatures -input $trainFile -makeDomain $makeDomain -featureCountThreshold $featureCountThreshold  -domain $domainName -output $output $featureSpec -lengthRound $lengthRounding -pad $pad 

$makeFeatures -input $trainFile -makeDomain $makeDomain -featureCountThreshold $featureCountThreshold  -domain $domainName -output $output $featureSpec -lengthRound $lengthRounding -pad $pad 


makeDomain=0
for f in $allFiles
do 
	file=`echo $f | cut -d":" -f1`
	dataset=`echo $f | cut -d":" -f2`
	output=$outDir/$dataset.int.all

    lenRound=0
	if [ "$dataset" == "train" ]; then
		lenRound=$lengthRounding
	fi
	lengthArgs="-lengthRound $lenRound"

	echo making features for $dataset
	#this extracts features and writes out an intermediate ascii int
	echo 	$makeFeatures -input $file -makeDomain $makeDomain -domain $domainName -output $output -pad $pad $lengthArgs $featureSpec $lengthArgs

	$makeFeatures -input $file -makeDomain $makeDomain -domain $domainName -output $output -pad $pad $lengthArgs $featureSpec $lengthArgs

	outDirForDataset=$outDir/$dataset
	outNameForDataset=$outDirForDataset/$dataset-
	mkdir -p $outDirForDataset
	echo splitting $dataset by length
	outSuffix=.int
	#this splits the ascii int file into separate files where each file contains examples of the same length
	$splitByLength $output $outNameForDataset $outSuffix

	rm -f $outDir/$dataset.list #the downstream training code reads in this list of filenames of the data files, split by length
	echo converting $dataset to torch files
	for ff in $outNameForDataset*.int 
	do
		out=`echo $ff | sed 's|.int$||'`.torch
		
		$int2torch -input $ff -output $out -tokenLabels $tokLabels -tokenFeatures $tokFeats $addOne #convert to torch format
		echo $out >> $outDir/$dataset.list
	done

done




