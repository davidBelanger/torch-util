##specifications about the input data
trainFile=simpleData/train.txt.small
devFile=simpleData/dev.txt.small
testFile=simpleData/test.txt.small
tokLabels=0 #whether the input has labels at the token level (alternative: at the sentence level)

##specification about features
tokFeats=0 #whether to use features for each token (alternative: just token string)
featureTemplates=tokenString,isCap,isNumeric #if using token features, this is a list of the names of the templates to use (assuming that each of these is implemented in $makeFeatures)

##parameters to choose
featureCountThreshold=5
minLength=5 #this ensures that the output is at least this many tokens
lengthRounding=5 #this pads such that every sequence has a length that is a multiple of <lengthRounding> (only used on train data)
pad=2 #this puts <pad> dummy tokens on each side

outDir=proc/
name=debug #name for the expt
domainName=$outDir/domain

#script paths
makeFeatures="python featureExtraction.py"
splitByLength="python splitByLength.py"
int2torch="th int2torch.lua"

allFiles="$trainFile:train $devFile:dev $testFile:test"



##first, process just the train data, in order to establish string->int mappings for the feature templates
makeDomain=1
output="/dev/null"
$makeFeatures -input $trainFile -makeDomain $makeDomain -featureCountThreshold $featureCountThreshold  -pad $pad -domain $domainName -output $output -tokenFeatures $tokFeats -featureTemplates $featureTemplates


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
	echo $makeFeatures -input $file -makeDomain $makeDomain -domain $domainName -output $output -pad $pad $lengthArgs -tokenFeatures $tokFeats -featureTemplates $featureTemplates $lengthArgs

	$makeFeatures -input $file -makeDomain $makeDomain -domain $domainName -output $output -pad $pad $lengthArgs -tokenFeatures $tokFeats -featureTemplates $featureTemplates $lengthArgs

	outDirForDataset=$outDir/$dataset
	outNameForDataset=$outDirForDataset/$dataset-
	mkdir -p $outDirForDataset
	echo splitting $dataset by length
	outSuffix=.int
	$splitByLength $output $outNameForDataset $outSuffix

	rm -f $outDir/$dataset.list
	echo converting $dataset to torch files
	for ff in $outNameForDataset*.int 
	do
		out=`echo $ff | sed 's|.int$||'`.torch
		$int2torch -input $ff -output $out -tokenLabels $tokLabels -tokenFeatures $tokFeats
		echo $out >> $outDir/$dataset.list
	done

done




