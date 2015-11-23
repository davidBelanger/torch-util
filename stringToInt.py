import fileinput
import sys
from collections import defaultdict


def main():
	dict = {}
	count = 0
	out = open(sys.argv[2], 'w')
	outmapping = open(sys.argv[3], 'w')


	for line in fileinput.input(sys.argv[1]):
		line = line.rstrip()
		if(line not in  dict):
			count = count +1
			dict[line] = count
		print >> out, dict[line]	

	for key, value in sorted(dict.iteritems(), key=lambda x: x[1]):
		print >> outmapping, "{}\t{}".format(key,value)

if __name__ == "__main__":
    main()

