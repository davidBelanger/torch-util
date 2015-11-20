import argparse
import fileinput
import sys

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("-trainFile",type=str,help="input")
	parser.add_argument("-devFile",type=str,help="output")
	parser.add_argument("-testFile",type=str,help="output")
	parser.add_argument("-name",type=str,help="output")
	parser.add_argument("-outDir",type=str,help="output")

	parser.add_argument("-tokLabels",type=str,help="output")
	parser.add_argument("-tokFeats",type=str,help="output")
	parser.add_argument("-featureTemplates",type=str,help="output")
	parser.add_argument("-featureCountThreshold",type=str,help="output")
	parser.add_argument("-lengthRounding",type=str,help="output")
	parser.add_argument("-pad",type=str,help="output")
	#todo: something where it forces everything to have the same length

	args = parser.parse_args()
	args = vars(args)
	#script paths

	files = {}
	assert(args.trainFile not null) #todo finish
	if(args.) #todo: build up a dictionary of the non-empty files

	#TODO: take the torch-util path as a command line arg, so that everything can be called from a different directory. instead, can you grab the dirname for this executable?
	args['makeFeatures']="python featureExtraction.py"
	args['addOne']="-addOne 1" #use this if preprocessing is 0-indexed. set this to the empty string if your preprocessing is 1-indexed. (featureExtraction.py is 0-indexed)
	args['splitByLength']="python splitByLength.py"
	args['int2torch']="th int2torch.lua"

	args["domainName"]="%(outDir)s/domain" % args

	args["featureSpec"]="-tokenFeatures $(tokFeats)s  -tokenLabels %(tokLabels)s -featureTemplates %(featureTemplates)s" % args

#---------------Below this are things that you shouldn't need to change-------------------#

	##first, establish string->int mappings for the feature templates
	args["makeDomain"]=1
	args["output"]="/dev/null"

	cmd = "%(makeFeatures)s -input %(trainFile)s -makeDomain %(makeDomain)s -featureCountThreshold %(featureCountThreshold)s  -domain %(domainName)s -output %(output)s %(featureSpec)s -lengthRound %(lengthRounding)s -pad %(pad)s " % args
	print(cmd)
	#todo: system call

makeDomain=0
for f in files
do 
	file=`echo $f | cut -d":" -f1`
	dataset=`echo $f | cut -d":" -f2`
	output=$outDir/$dataset.int.all

    lenRound=0
	if [ "$dataset" == "train" ]; then
		lenRound=lengthRounding
	fi
	lengthArgs="-lengthRound %(lenRound)s" % args

	print("making features for %s".format(dataset))
	#this extracts features and writes out an intermediate ascii int
	cmd = "%(makeFeatures)s -input %(file)s -makeDomain %(makeDomain)s -domain %(domainName)s -output %(output)s -pad %(pad)s %(lengthArgs)s %(featureSpec)s %(lengthArgs)s" % args

	outDirForDataset="%(outDir)s/%(dataset)s" % args
	outNameForDataset="%(outDirForDataset)s/%(dataset)s-" % args
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


if __name__ == "__main__":
    main()


