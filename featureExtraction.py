import argparse
import fileinput
from featureTemplate import * 


#you may want to change these
def tokenize(sentence):
	strings = sentence.split(" ")
	return strings

def normalize(string):
	return string #todo: change this

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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-input",type=str,help="input")
	parser.add_argument("-output",type=str,help="output")

	parser.add_argument("-makeDomain",type=int,help="whether to make domain or to write out int feature files")
	parser.add_argument("-featureCountThreshold",type=int,help="threshold for considering features")
	parser.add_argument("-pad",type=int,help="how much to pad the input on each side")
	parser.add_argument("-domain",type=str,help="basename for the domain files")
	parser.add_argument("-tokenFeatures",type=int,help="whether to use token features")
	parser.add_argument("-featureTemplates",type=str,help="comma-separated list of the names of the feature templates to use")

	args = parser.parse_args()

	makeDomain = args.makeDomain

	featureTemplateFunctions = [TokenString(),Capitalized()]
	featureTemplates = FeatureTemplates(args.tokenFeatures,featureTemplateFunctions,args.featureCountThreshold)
	if(not makeDomain):
		featureTemplates.loadDomains(args.domain)

	for line in fileinput.input(args.input):
		fields = line.split("\t")
		label = fields[0]
		text = fields[1].rstrip()
		toks = tokenize(text)
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



if __name__ == "__main__":
    main()
