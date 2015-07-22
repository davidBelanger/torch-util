from collections import defaultdict
import operator
import json
import re

nlpFeatureConstants = {
	"oov" : "#UNK",
	"padleft" : "#PadLeft"  ,
	"padright" : "#PadRight" ,
	"special": "#Special" 
}

##***this leaves self.featureFunction as abstract. when creating instances of FeatureTemplate, implement this
##**it's also expected that you proved a string self.name

class FeatureTemplate:

	#skipSpecialChars = False # the default is to add a special feature if you see a special character
	spec = re.compile('^#')
	useSpecialWords = False


	def __init__(self,allowOOV):
		self.buildCounts = True
		self.counts = defaultdict(int)
		self.domain = None
		self.assertInDomain = not allowOOV

	def isSpecial(self,tokStr):
		return self.spec.match(tokStr)

	def extractFeature(self,normalizedString):
		feat = None
		if(self.isSpecial(normalizedString)):
			if(self.useSpecialWords):
				feat = normalizedString 
			else:
				feat = nlpFeatureConstants["special"]
		else:
			feat = self.featureFunction(normalizedString)
	
		if(self.buildCounts):
			self.counts[feat]+= 1 
		return feat

	def writeDomain(self,file):
		print "writing: " + file

		data = {
			'name':self.name,
			'domain' : self.domain
		}
		with open(file, 'w') as outfile:
   			json.dump(data, outfile)


	def constructDomain(self,featureCountThreshold):
		filteredKeys = {k: v for k, v in self.counts.iteritems() if v > featureCountThreshold}
		sortedKeysByFrequency =  sorted(filteredKeys.items(),key = operator.itemgetter(1),reverse=True)
		self.domain = dict(map (lambda t: (t[1], t[0]), enumerate( map (lambda x: x[0], sortedKeysByFrequency)))) ##map from key to index
		if(not self.assertInDomain):
			self.domain[nlpFeatureConstants["oov"]] = len(self.domain)

	def convertToInt(self,feat):
		if(feat in self.domain):
			return self.domain[feat]
		else:
			assert not self.assertInDomain, "input value " + feat + " not in domain" + " name = "  + self.name
			return self.domain[nlpFeatureConstants["oov"]]



	def loadDomain(self,file):
		with open(file, 'r') as datfile:
   			data = json.load(datfile)
   			assert data['name'] == self.name
    		self.domain = data['domain']




class FeatureTemplates:


	def __init__(self,useTokenFeatures,featureTemplates,featureCountThreshold):
		self.candidateTemplates = []
		self.featureTemplates = featureTemplates
		self.useFeats = useTokenFeatures

		self.featureCountThreshold = featureCountThreshold

		for d in self.featureTemplates:
			d.buildCounts = True

	def convertToInt(self,tokStringFeats):
		return map(lambda x: x[1].convertToInt(x[0]),zip(tokStringFeats,self.featureTemplates))


	def loadDomains(self,domainFileBase): 
		for template in self.featureTemplates:
			template.loadDomain(domainFileBase + "." + template.name)


	def writeDomains(self,domainFileBase): 
		for template in self.featureTemplates:
			fn = domainFileBase + "." + template.name
			template.writeDomain(fn)

	def constructDomains(self):
		for d in self.featureTemplates:
			d.constructDomain(self.featureCountThreshold)
			d.buildCounts = False

	def extractFeatures(self,normalizedString):
		return map(lambda x: x.extractFeature(normalizedString),self.featureTemplates)


	def convertFeaturesForPrinting(self,sentenceFeatures):
		if(not self.useFeats):
			return " ".join(map(lambda x: str(x[0]),sentenceFeatures))
		else:
			return " ".join(map(lambda x: ",".join(map(lambda y: str(y),x)),sentenceFeatures))


