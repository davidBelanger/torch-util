import json

with open('proc4/domain' + ".tokenString") as data_file:
	tokenStringDomain = json.load(data_file)["domain"]
	print tokenStringDomain.keys()

