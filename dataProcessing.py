import argparse
import fileinput
import sys
import os
import glob
from subprocess import check_call

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("-trainFile",required=True,type=str)
	parser.add_argument("-devFile",type=str)
	parser.add_argument("-testFile",type=str)
	parser.add_argument("-name",required=True,type=str)
	parser.add_argument("-outDir",required=True,type=str)

	parser.add_argument("-tokLabels",type=int,default="0")
	parser.add_argument("-tokFeats",type=int,default="0")
	parser.add_argument("-featureTemplates",type=str,default="tokenString,isCap")
	parser.add_argument("-featureCountThreshold",type=str,default=5)
	parser.add_argument("-lengthRounding",type=int,default=0)
	parser.add_argument("-pad",type=int,default=0)

	parser.add_argument("-verbose",type=int,default=0)

	#todo: something where it forces everything to have the same length

	args = parser.parse_args()
	args = vars(args)

	files = {}
	assert(args["trainFile"]  != None)
	files["train"] = args["trainFile"]
	if(args["devFile"] != None):
		files["dev"] = args["devFile"]

	if(args["testFile"] != None): 
		files["test"] = args["testFile"]

	scriptPath = os.path.dirname(sys.argv[0]) 
	scriptPath = scriptPath if (scriptPath != "") else "."
	args['makeFeatures']="python {}/featureExtraction.py".format(scriptPath)
	args['addOne']="-addOne 1" #use this if preprocessing is 0-indexed. set this to the empty string if your preprocessing is 1-indexed. (featureExtraction.py is 0-indexed)
	args['splitByLength']="python {}/splitByLength.py".format(scriptPath)
	args['int2torch']="th {}/int2torch.lua".format(scriptPath)
	args["domainName"]="%(outDir)s/domain" % args
	args["featureSpec"]="-tokenFeatures %(tokFeats)s  -tokenLabels %(tokLabels)s -featureTemplates %(featureTemplates)s" % args

	verbose = args["verbose"]
	def syscall(cmd):
		if(verbose):
			print(cmd)
		check_call(cmd,shell=True)

	##first, establish string->int mappings for the feature templates
	args["makeDomain"]=1
	args["output"]="/dev/null"
	cmd = "%(makeFeatures)s -input %(trainFile)s -makeDomain %(makeDomain)s -featureCountThreshold %(featureCountThreshold)s  -domain %(domainName)s -output %(output)s %(featureSpec)s -lengthRound %(lengthRounding)s -pad %(pad)s " % args
	syscall(cmd)


	outDir=args["outDir"]
	args["makeDomain"]=0

	for dataset,file in files.iteritems():
		output="{}/$dataset.int.all".format(outDir)
		args["output"] = output
		lenRound=0
		if(dataset == "train"):
			lenRound=args["lengthRounding"]

		args["lengthArgs"]="-lengthRound {}".format(lenRound)

		print("making features for {}".format(dataset))
		args["file"] = file
		args["dataset"] = dataset
		#this extracts features and writes out an intermediate ascii int
		cmd = "%(makeFeatures)s -input %(file)s -makeDomain %(makeDomain)s -domain %(domainName)s -output %(output)s -pad %(pad)s %(lengthArgs)s %(featureSpec)s %(lengthArgs)s" % args
		syscall(cmd)


		args["outDirForDataset"]="%(outDir)s/%(dataset)s" % args
		args["outNameForDataset"]="%(outDirForDataset)s/%(dataset)s-" % args
		cmd = "mkdir -p {}".format(args["outDirForDataset"])
		syscall(cmd)

		print("splitting {} by length".format(dataset))
		args["outSuffix"]=".int"
		#this splits the ascii int file into separate files where each file contains examples of the same length
		cmd="%(splitByLength)s %(output)s %(outNameForDataset)s %(outSuffix)s" % args
		syscall(cmd)

		#todo: system

		cmd=="rm -f {}/{}.list".format(outDir,dataset) #the downstream training code reads in this list of filenames of the data files, split by length
		syscall(cmd)

		#todo: system
		print("converting {} to torch files".format(dataset))
		
		for ff in glob.glob('{}*.int'.format(args["outNameForDataset"])):
			args["ff"] = ff
			cmd="out=`echo %(ff)s | sed 's|.int$||'`.torch;%(int2torch)s -input %(ff)s -output $out -tokenLabels %(tokLabels)s -tokenFeatures %(tokFeats)s %(addOne)s;echo $out >> %(outDir)s/%(dataset)s.list" % args
			syscall(cmd)


if __name__ == "__main__":
	main()


