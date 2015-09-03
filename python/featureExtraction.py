import argparse
import fileinput
from featureTemplate import * 
import re
import json

class TokenString(FeatureTemplate):
	name = 'tokenString'
	useSpecialWords = True
	def featureFunction(self,normalizedString):
		return normalizedString

class Capitalized(FeatureTemplate):
	name = 'isCap'
	def featureFunction(self,normalizedString):
		if(normalizedString[0].isupper()):
			return "1"
		else:
			return "0"

class IsNumeric(FeatureTemplate):
	name = 'isNumeric'
	numMatch = re.compile("^(#NUM)+$")

	def featureFunction(self,normalizedString):
		if self.numMatch.match(normalizedString):
			return "1"
		else:
			return "0"

class Suffix(FeatureTemplate):
	def __init__(self,width,allowOOV):
		self.width = width
		self.name = "Suffix-"+str(width)
		FeatureTemplate.__init__(self,allowOOV)

	def featureFunction(self,normalizedString):
		return normalizedString[max(0,len(normalizedString) - self.width) : len(normalizedString)]

class Prefix(FeatureTemplate):
	def __init__(self,width,allowOOV):
		self.width = width
		self.name = "Prefix-"+str(width)
		FeatureTemplate.__init__(self,allowOOV)


	def featureFunction(self,normalizedString):
		return normalizedString[0 : min(len(normalizedString),self.width)]


class Label(FeatureTemplate):
	name = 'label'
	useSpecialWords = True

	def featureFunction(self,label):
		return label

def getTemplates(args):
	if(not args.tokenFeatures):
		return [TokenString(allowOOV = True)]
	else:
		templates = []
		for name in args.featureTemplates.split(","):
			if(name == "tokenString"):
				templates.append(TokenString(allowOOV = True))
			elif(name == "isCap"):
				templates.append(Capitalized(allowOOV = False))
			elif(name == "isNumeric"):
				templates.append(IsNumeric(allowOOV = False))
			elif(re.match(r"Suffix-\d+",name)):
				num = re.replace(r"Suffix-","",name)
				templates.append(Suffix(int(num),allowOOV = True))
			elif(re.match(r"Prefix-\d+",name)):
				num = re.replace(r"Prefix-","",name)
				templates.append(Prefix(int(num),allowOOV = True))
		return templates

#you may want to change these
def tokenize(sentence):
	strings = sentence.split(" ")
	return strings

num = re.compile("\d")
def normalize(string):
	string = re.sub(num,"#NUM",string)
	return string 


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-input",type=str,help="input")
	parser.add_argument("-output",type=str,help="output")
	parser.add_argument("-domain",type=str,help="basename for the domain files")

	parser.add_argument("-makeDomain",type=int,help="whether to make domain or to write out int feature files")

	parser.add_argument("-tokenLabels",type=int,help="whether annotation is at the token level (vs. sentence level)",default=0)
	
	parser.add_argument("-featureCountThreshold",type=int,help="threshold for considering features",default=0)
	parser.add_argument("-tokenFeatures",type=int,help="whether to use token features",default=0)
	parser.add_argument("-featureTemplates",type=str,help="comma-separated list of the names of the feature templates to use",default="tokenString,isCap,isNumeric")

	parser.add_argument("-pad",type=int,help="how much to pad the input on each side",default=0)
	parser.add_argument("-lengthRound",type=int,help="pad such that all token sequences have length that is a multiple of lengthRound")
	
	#parser.add_argument("-minLength",type=int,help="minimum length of an observation sequence (after padding).")
	
	args = parser.parse_args()

	makeDomain = args.makeDomain

	featureTemplateFunctions = getTemplates(args)

	featureTemplates = FeatureTemplates(args.tokenFeatures,featureTemplateFunctions,args.featureCountThreshold)

	labelDomain = Label(allowOOV = False) 

	out = None
	if(not makeDomain):
		featureTemplates.loadDomains(args.domain)
		labelDomain.loadDomain(args.domain + "-label")
		out = open(args.output, 'w')
	else:
		labelDomain.buildCounts = True

	tokenLabels = args.tokenLabels == 1
	for line in fileinput.input(args.input):
		fields = line.split("\t")
		labelString = fields[0]
		text = fields[1].rstrip()
		toks = tokenize(text)

		labels = None
		if(tokenLabels): 
			labels = labelString.split(" ")

	    #this pads the data such that both the token and label sequences have length that's a multiple of lengthRound
		if(args.lengthRound > 0):
			toks = addPaddingForLengthRounding(toks,args.lengthRound,nlpFeatureConstants["padleft"],nlpFeatureConstants["padright"])
			labels = addPaddingForLengthRounding(labels,args.lengthRound,nlpFeatureConstants["padleft"],nlpFeatureConstants["padright"])

		#this pads the tokens, but not the labels. this is useful when using CNNs
		if(args.pad > 0):
			toks = addPadding(toks,args.pad,nlpFeatureConstants["padleft"],nlpFeatureConstants["padright"])

		normalizedToks = map(lambda st: normalize(st), toks)
		stringFeatures = map(lambda tok: featureTemplates.extractFeatures(tok), normalizedToks)

		if(not makeDomain):
			intFeatures = map(lambda tokStringFeats: featureTemplates.convertToInt(tokStringFeats), stringFeatures)
			intLabel = None
			if(tokenLabels):
				intLabel = " ".join(map(lambda l: str(labelDomain.convertToInt(l)),labels))
			else:
				intLabel = str(labelDomain.convertToInt(labelString))
			print >> out, "{0}\t{1}".format(intLabel,featureTemplates.convertFeaturesForPrinting(intFeatures))
		else:
			if(not tokenLabels):
				labelDomain.extractFeature(labelString) #this is for adding label to the domain
			else:
				ff = map(lambda l: labelDomain.extractFeature(l),labels)

	if(makeDomain):
		print("finished processing text. Now constructing domains")
		featureTemplates.constructDomains()
		print('writing domain files')
		featureTemplates.writeDomains(args.domain)
		labelDomain.constructDomain(0)
		labelDomain.writeDomain(args.domain + "-label")
		writeAsciiDomainInfo(args.domain,featureTemplates,labelDomain)
		with open(args.domain + ".tokenString") as data_file:
			tokenStringDomain = json.load(data_file)["domain"]
			writeAsciiList(args.domain + "-vocab.ascii",tokenStringDomain.keys())


		writeAsciiList(args.domain + "-labels.ascii",labelDomain.domain.keys())
	else:
		print "wrote " + args.output
		out.close()

def writeAsciiList(outFile,list):
	out = open(outFile,'w')
	for item in list:
  		out.write("%s\n" % item)
  	out.close()


def writeAsciiDomainInfo(domainFileName,featureTemplates,labelDomain):
	fn = domainFileName + ".domainSizes.txt"
	print 'writing observation domain size info ' + fn
	out = open(fn, 'w')
	for template in featureTemplates.featureTemplates:
		name = template.name
		size = len(template.domain) 
		print >> out, name + "\t" + str(size)

	out.close()

	fn = domainFileName + ".labelDomainSize.txt"
	print 'writing label domain size info ' + fn
	out = open(fn, 'w')
	print >> out, str(len(labelDomain.domain))
	out.close()

def addPaddingForLengthRounding(toks,targetLengthDivider,leftStr,rightStr):
	length = len(toks)
	targetLength = length - (length % targetLengthDivider) + targetLengthDivider #this rounds up to the nearest multiple of targetLengthDivider

	addToFront = False
	while(len(toks) < targetLength):
		if(addToFront):
			toks.insert(0,leftStr)
		else:
			toks.append(rightStr)	
		addToFront = not addToFront
	return toks

def addPadding(toks,pad,leftStr,rightStr):
	for i in range(0,pad):
		toks.insert(0,leftStr)
		toks.append(rightStr)
	return toks

if __name__ == "__main__":
    main()
