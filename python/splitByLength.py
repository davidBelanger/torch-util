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
	for line in fileinput.input(inputFile):
		line = line.rstrip()
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