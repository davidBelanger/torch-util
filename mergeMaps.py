import fileinput
import sys

##NOTE: this could be more flexible, by being less sensitivie to the order of the input files. Right now, it assumes that both files have the same order of keys

def loadList(file):
	H = []
	for line in fileinput.input(file):
		line = line.rstrip()
		fields = line.split("\t")
		H.append(fields)
	return H

def main():

	h1 = loadList(sys.argv[1])
	h2 = loadList(sys.argv[2])
	cnt = 0
	n = len(h1)
	for i in range(0,n):
		assert(h1[i][0] == h2[i][0])
		print(h1[i][0] + "\t" + h1[i][1] + "\t" + h2[i][1])

if __name__ == "__main__":
    main()