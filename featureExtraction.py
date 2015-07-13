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

##these are the actual feature templates

class Capitalized(FeatureTemplate):
	name = 'isCap'
	def featureFunction(self,normalizedString):
		if(normalizedString[0].isupper()):
			return "1"
		else:
			return "0"

class TokenString(FeatureTemplate):
	name = 'tokenString'
	def featureFunction(self,normalizedString):
		return normalizedString

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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-input",type=str,help="input")
	parser.add_argument("-output",type=str,help="output")
	parser.add_argument("-domain",type=str,help="basename for the domain files")

	parser.add_argument("-makeDomain",type=int,help="whether to make domain or to write out int feature files")
	
	parser.add_argument("-featureCountThreshold",type=int,help="threshold for considering features")
	parser.add_argument("-tokenFeatures",type=int,help="whether to use token features")
	parser.add_argument("-featureTemplates",type=str,help="comma-separated list of the names of the feature templates to use")

	parser.add_argument("-pad",type=int,help="how much to pad the input on each side")
	
	#todo: add these eventually. 
	#parser.add_argument("-minLength",type=int,help="minimum length of an observation sequence (after padding).")
	#parser.add_argument("-lengthRound",type=int,help="pad such that all token sequences have length that is a multiple of lengthRound")
	
	args = parser.parse_args()

	makeDomain = args.makeDomain

	featureTemplateFunctions = [TokenString()]
	if(args.tokenFeatures):
		featureTemplateFunctions.append(Capitalized())
		featureTemplateFunctions.append(IsNumeric())
		featureTemplateFunctions.append(Suffix(2))
		featureTemplateFunctions.append(Prefix(3))


		#todo: add more, and use featureTemplateNames...

	featureTemplates = FeatureTemplates(args.tokenFeatures,featureTemplateFunctions,args.featureCountThreshold)
	if(not makeDomain):
		featureTemplates.loadDomains(args.domain)

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
			print "{0}\t{1}".format(label,featureTemplates.convertFeaturesForPrinting(intFeatures))

	if(makeDomain):
		print("finished processing text. Now constructing domains")
		featureTemplates.constructDomains()
		print('writing domain files')
		featureTemplates.writeDomains(args.domain)

def addPadding(toks,pad,leftStr,rightStr):
	for i in range(0,pad):
		toks.insert(0,leftStr)
		toks.append(rightStr)

if __name__ == "__main__":
    main()
