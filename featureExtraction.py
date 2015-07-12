import argparse
import fileinput
from collections import defaultdict
import operator
import json


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
	featureTemplates = FeatureTemplates(args.tokenFeatures,args.featureTemplates,args.featureCountThreshold)
	if(not makeDomain):
		featureTemplates.loadDomain(args.domain)

	for line in fileinput.input(args.input):
		fields = line.split("\t")
		label = fields[0]
		text = fields[1].rstrip()
		toks = tokenize(text)
		print(toks)
		stringFeatures = map(lambda tok: featureTemplates.extractFeatures(tok), toks)
		if(not makeDomain):
			intFeatures = map(lambda tokStringFeats: featureTemplates.convertToInt(tokStringFeats), stringFeatures)
			print "{0}\t{1}".format(label,featureTemplates.convertFeaturesForPrinting(intFeatures))

	if(makeDomain):
		print("finished processing text. Now constructing domains")
		featureTemplates.constructDomains()
		print('writing domain files')
		featureTemplates.writeDomains(args.domain)



class FeatureTemplates:

	candidateTemplates = []
	featureTemplates = []
	def __init__(self,useTokenFeatures,featureTemplateNames,featureCountThreshold):
		self.useFeats = useTokenFeatures
		if(self.useFeats):
			self.featureTemplateNames = featureTemplateNames.split(",")
		self.featureCountThreshold = featureCountThreshold
		self.featureTemplates.append(TokenString())

		#todo: need to use the featureTemplateNames variable

		for d in self.featureTemplates:
			d.buildCounts = True

	def convertToInt(self,tokStringFeats):
		return map(lambda x: x[1].convertToInt(x[0]),zip(tokStringFeats,featureTemplates))


	def loadDomains(self,domainFileBase): 
		for template in self.featureTemplates:
			template.loadDomain(domainFileBase + "." + template.name)


	def writeDomains(self,domainFileBase): 
		for template in self.featureTemplates:
			template.writeDomain(domainFileBase + "." + template.name)

	def constructDomains(self):
		for d in self.featureTemplates:
			d.constructDomain(self.featureCountThreshold)
			d.buildCounts = false

	def extractFeatures(self,tokenString):
		normalizedString = tokenString #todo: change
		return map(lambda x: x.extractFeature(normalizedString),self.featureTemplates)


	def convertFeaturesForPrinting(self,sentenceFeatures):
		if(not useFeats):
			return " ".join(map(lambda x: x[0],sentenceFeatures))
		else:
			return " ".join(map(lambda x: ",".join(x),sentenceFeatures))


class FeatureTemplate:
	def __init__(self):
		self.buildCounts = True
		self.counts = defaultdict(int)
		self.domain = None


	def extractFeature(self,normalizedString):
		feat = self.featureFunction(normalizedString)
		print feat
		if(self.buildCounts):
			self.counts[feat]+= 1 
		return feat



	def writeDomain(self,file):
		data = {
			name:self.name,
			domain : self.domain
		}
		with open(file, 'w') as outfile:
   			json.dump(data, outfile)



	def constructDomain(self,featureCountThreshold):
		self.domain =  sorted({k: v for k, v in self.counts.iteritems() if v > featureCountThreshold},key=lambda x: x[1])
		self.domain[defaultValue] = len(self.domain) + 1


    def convertToInt(self,feat):
    	if(feat in self.domain):
    		return self.domain[feat]
    	else:
    		return self.domain[defaultValue]



	def loadDomain(self,file):
		with open(file, 'r') as datfile:
   			data = json.load(datfile)
   			assert data.name == self.name
    		self.domain = data.domain


##these are the actual feature templates
class TokenString(FeatureTemplate):
	name = 'tokenString'
	def featureFunction(self,normalizedString):
		return normalizedString


if __name__ == "__main__":
    main()
