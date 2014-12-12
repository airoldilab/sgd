'''
Read in the first 20000 lines of covtype.data

Prerequisite:
- have covtype.data in the same folder
'''

with open("covtype.data") as myfile:
    head = [next(myfile) for x in xrange(20000)]

with open("covtype_simple.data", "w") as myfile:
    for line in head:
        myfile.write(line)


