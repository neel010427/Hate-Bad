#from pymongo import MongoClient
import json

#client = MongoClient('mongodb+srv://haspburn71280:H8IsNoGood@hatebaddb-kbv0e.gcp.mongodb.net/test?retryWrites=true&w=majority')
#db = client.hatebad
#test = db.trainingset

allFileData = dict()

with open('metadata.csv', 'r') as info:
	line = info.readline()
	line = info.readline()
	i = 0
	while line:
		data = line.strip().split(',')
		print(data[0])
		allFileData[data[0]] = data[4]
		i = i + 1
		line = info.readline()

print(allFileData)

with open('hateFileData.json', 'w') as f:
	json.dump(allFileData, f)
