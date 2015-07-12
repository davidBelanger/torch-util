##specifications about the input data
trainFile=TODO
devFile=TODO
testFile=TODO
tokLabels=TODO #whether the input has labels at the token level (alternative: at the sentence level)

##specification about features
tokFeats=1 #whether to use features for each token (alternative: just token string)
featureTemplates=TODO #if using token features, this is a list of the names of the templates to use (assuming that each of these is implemented in $makeFeatures)

##parameters to choose
featureCountThreshold=5
minLength=5 #this ensures that the output is at least this many tokens
lengthRounding=0 #this pads such that every sequence has a length that is a multiple of <lengthRounding> (typically only used on train data)
pad=2 #this puts <pad> dummy tokens on each side

outDir=TODO
name=TODO #name for the expt
domainDir=$outDir
domainName=$domainDir/$name

#script paths
makeFeatures=TODO
splitByLength="perl splitByLength.pl"

allFiles="$trainFile:train $devFile:dev $testFile:test"



##first, process just the train data, in order to establish string->int mappings for the feature templates
makeDomain=1
output=""
$makeFeatures -input $trainFile -makeDomain $makeDomain -featureCountThreshold $featureCountThreshold  -pad $pad -domain $domainName -output $output -tokenFeatures $tokFeats -featureTemplates $featureTemplates


makeDomain=0
for f in $allFiles
do 
	file=`echo $f | cut -d":" -f1`
	dataset=`echo $f | cut -d":" -f2`
	output=$outDir/$dataset.int.all

    lenRound=$lengthRounding
	if [ "$dataset" != "train" ]; then
		lenRound=0
	fi
	$makeFeatures -input $file -makeDomain $makeDomain -domain $domainName -output $output -pad $pad -minLength $minLength -lengthRound $lenRound -tokenFeatures $tokFeats -featureTemplates $featureTemplates

	outDirForDataset=$outDir/$dataset/
	mkdir -p $outDirForDataset
	$splitByLength $output $outDirForDataset

	rm -f $outDir/$dataset.list
	for ff in `find $outDirForDataset -type f ! -size 0` 
	do
		$int2torch -input $ff -output $ff.torch -tokenLabels $tokLabels -tokenFeatures $tokFeats
		echo $ff.torch >> $outDir/$dataset.list
	done

done



