import argparse
import fileinput



#todo: make a feature templates object 
#todo: make a feature template object

API: 
loadDomains(outBase)
writeDomains(outBase)
increment(stringFeatures)
constructDomains()
convertToInt(stringFeatures)
extractFeatures(tokenString)


#this extracts per-token features. it returns a function to be applied to a token string


#you may want to change this
def tokenize(sentence):
	strings = sentence.split(" ")
	return strings

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
	featureTemplates = FeatureTemplates(args.tokenFeatures,args.featureTemplates)
	if(not makeDomain):
		featureTemplates.loadDomain(args.domain)

	for line in fileinput.input(args.input):
		fields = line.split("\t")
		label = fields[0]
		text = fields[1].rstrip()
		toks = tokenize(text)
		stringFeatures = featureTemplates.extractFeatures(toks)
		if(not makeDomain):
			intFeatures = featureTemplates.convertToInt(stringFeatures)
			print "{0}\t{1}".format(label,convertFeaturesForPrinting(intFeatures))
		else:
			featureDomains.increment(stringFeatures)

	if(makeDomain):
		print("finished processing text. Now constructing domains")
		featureTemplates.constructDomains()
		print('writing domain files')
		featureTemplates.writeDomains(args.domain)










#this converts the mapped features so that they can be written out
def convertFeaturesForPrinting(sentenceFeatures):
	return " ".join(map(lambda x: ",".join(x),sentenceFeatures)


if __name__ == "__main__":
    main()
