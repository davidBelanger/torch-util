import argparse
import fileinput
from featureTemplate import * 
import re

#you may want to change these
def tokenize(sentence):
	strings = sentence.split(" ")
	return strings

num = re.compile("\d")
def normalize(string):
	string = re.sub(num,"#NUM",string)
	return string 

class TokenString(FeatureTemplate):
	name = 'tokenString'
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
	def __init__(self,width):
		self.width = width
		self.name = "Suffix-"+str(width)
		FeatureTemplate.__init__(self)

	def featureFunction(self,normalizedString):
		return normalizedString[max(0,len(normalizedString) - self.width) : len(normalizedString)]

class Prefix(FeatureTemplate):
	def __init__(self,width):
		self.width = width
		self.name = "Prefix-"+str(width)
		FeatureTemplate.__init__(self)


	def featureFunction(self,normalizedString):
		return normalizedString[0 : min(len(normalizedString),self.width)]


class Label(FeatureTemplate):
	name = 'label'
	def featureFunction(self,label):
		return label

def getTemplates(tmpltList):
	templates = []
	for name in tmpltList.split(","):
		if(name == "tokenString"):
			templates.append(TokenString())
		elif(name == "isCap"):
			templates.append(Capitalized())
		elif(name == "isNumeric"):
			templates.append(IsNumeric())
		elif(re.match(r"Suffix-\d+",name)):
			num = re.replace(r"Suffix-","",name)
			templates.append(Suffix(int(num)))
		elif(re.match(r"Prefix-\d+",name)):
			num = re.replace(r"Prefix-","",name)
			templates.append(Prefix(int(num)))


	return templates

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-input",type=str,help="input")
	parser.add_argument("-output",type=str,help="output")
	parser.add_argument("-domain",type=str,help="basename for the domain files")

	parser.add_argument("-makeDomain",type=int,help="whether to make domain or to write out int feature files")
	
	parser.add_argument("-featureCountThreshold",type=int,help="threshold for considering features",default=0)
	parser.add_argument("-tokenFeatures",type=int,help="whether to use token features",default=0)
	parser.add_argument("-featureTemplates",type=str,help="comma-separated list of the names of the feature templates to use",default="tokenString,isCap,isNumeric")

	parser.add_argument("-pad",type=int,help="how much to pad the input on each side",default=0)
	
	#todo: add these eventually. 
	#parser.add_argument("-minLength",type=int,help="minimum length of an observation sequence (after padding).")
	#parser.add_argument("-lengthRound",type=int,help="pad such that all token sequences have length that is a multiple of lengthRound")
	
	args = parser.parse_args()

	makeDomain = args.makeDomain

	featureTemplateFunctions = getTemplates(args.featureTemplates)

	featureTemplates = FeatureTemplates(args.tokenFeatures,featureTemplateFunctions,args.featureCountThreshold)
	labelDomain = Label() 

	out = None
	if(not makeDomain):
		featureTemplates.loadDomains(args.domain)
		labelDomain.loadDomain(args.domain + "-label")
		labelDomain.assertInDomain = True
		out = open(args.output, 'w')
	else:
		labelDomain.buildCounts = True

	for line in fileinput.input(args.input):
		fields = line.split("\t")
		label = fields[0]
		text = fields[1].rstrip()
		toks = tokenize(text)
		if(args.pad > 0):
			toks = addPadding(toks,args.pad,nlpFeatureConstants["padleft"],nlpFeatureConstants["padright"])

		normalizedToks = map(lambda st: normalize(st), toks)
		stringFeatures = map(lambda tok: featureTemplates.extractFeatures(tok), normalizedToks)

		if(not makeDomain):
			intFeatures = map(lambda tokStringFeats: featureTemplates.convertToInt(tokStringFeats), stringFeatures)
			intLabel = labelDomain.convertToInt(label)
			print >> out, "{0}\t{1}".format(intLabel,featureTemplates.convertFeaturesForPrinting(intFeatures))
		else:
			labelDomain.extractFeature(label) #this is for adding label to the domain


	if(makeDomain):
		print("finished processing text. Now constructing domains")
		featureTemplates.constructDomains()
		print('writing domain files')
		featureTemplates.writeDomains(args.domain)
		labelDomain.constructDomain(0)
		labelDomain.writeDomain(args.domain + "-label")
		writeAsciiDomainInfo(args.domain,featureTemplates,labelDomain)
	else:
		print "wrote " + args.output
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

def addPadding(toks,pad,leftStr,rightStr):
	for i in range(0,pad):
		toks.insert(0,leftStr)
		toks.append(rightStr)
	return toks

if __name__ == "__main__":
    main()
