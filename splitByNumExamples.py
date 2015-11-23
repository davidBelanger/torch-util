import sys
import fileinput


def getLength(line):
	fields = line.split("\t")
	return len(fields[1].split(" "))

def main():
	inputFile = sys.argv[1]
	outputBase = sys.argv[2]
	outSuffix=sys.argv[3]

	handles = {}
	prevId = None
	for line in fileinput.input(inputFile):
		line = line.rstrip()
		fields = line.split("\t")
		id = fields[0]
		if(id != prevId):
			if(not first): 
				#todo: printstuff...
			curExamples = {}
		else:
			curExamples.append()

		length = getLength(line)
		out = None
		if(length in handles):
			out = handles[length]
		else:
			out = open(outputBase + "-"+str(length) + outSuffix, 'w')
			handles[length] = out

		print >> out, line


if __name__ == "__main__":
    main()