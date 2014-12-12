'''
Read in the first 20000 lines of covtype.data

Prerequisite:
- have covtype.data in the same folder
'''
import csv

with open("covtype.data") as myfile:
	data = [next(myfile) for x in xrange(20000)]

header = ['Elevation', 'Aspect', 'Slope', 'Horiz_To_Hydrology', 'Verti_To_Hydrology', \
	'Horiz_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horiz_To_FirePt']

for i in xrange(4):
	header += ['Wilderness_Area_' + str(i+1)]
for i in xrange(40):
	header += ['Soil_Type_' + str(i+1)]
header += ['Cover_Type']

with open("covtype_simple.csv", "w") as myfile:
	csv_writer = csv.writer(myfile, delimiter=',')
	csv_writer.writerow(header)
	for line in data:
		line_split = line.split(',')

		csv_writer.writerow([int(x) for x in line_split])


