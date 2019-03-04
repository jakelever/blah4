from intervaltree import Interval, IntervalTree

file1 = """
chr1	1000	2000	gain
chr1	3000	4000	gain
chr1	5000	6000	gain
"""

file2 = """
chr1	1500	2000	gain
chr1	3500	4000	gain
chr1	5500	6000	gain
"""

file3 = """
chr1	1000	4000	gain
chr1	5000	5100	gain
"""

trees = {'chr1':IntervalTree()}

print("Load the data from 'three' fake files into the IntervalTree for that specific chromosome")
for name,data in [('file1',file1), ('file2',file2), ('file3',file3)]:
	print("== %s ==" % name)
	for line in data.strip().split('\n'):
		print(line)
		chrom,start,end,info = line.strip().split('\t')
		start,end = int(start),int(end)

		# This saves the 'filename' to a particular region (start:end) in this tree.
		trees[chrom].addi(start,end,name)
	
print()
for chrom,tree in trees.items():
	print("Running for chromosome %s" % chrom)

	# We're going to use split_overlaps() to find all the possible subregions found in the file 
	# (so a region overlap of (1000 -> 2000) and (1500 -> 2500) becomes three subregions:
	#  1000 -> 1500, 1500 -> 2000 and 2000 -> 2500
	tree.split_overlaps()
	print("split_overlaps() gives these subregions:")
	print(tree)
	print()

	# Now we just get the sorted list of start and ends for all subregions
	startAndEnds = sorted(list(set([ (start,end) for start,end,val in tree ])))

	print("And if we count the number of intervals loaded from the file for each of these, we should get a reasonable output:")
	# Now we use the intervals found using split_overlaps on our original tree and count the number of intervals loaded from the files
	for start,end in startAndEnds:
		regionsOfInterest = tree[start:end]
		print(start,end,len(regionsOfInterest))



