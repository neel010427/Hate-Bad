from pymongo import MongoClient
import json
import os

with open('hateFileData.json', 'r') as f:
	tags = json.load(f)

##########################################################

testMetadata = list()

for filename in os.listdir('sampled_test'):
	with open('sampled_test/' + filename, 'r') as f:
		content = f.read()
		key = filename[0:-4]
		if tags[key] != "idk/skip":
			testMetadata.append({"text":content, "label":tags[key], "user":None})

#########################################################

trainMetadata = list()

for filename in os.listdir('sampled_train'):
	with open('sampled_train/' + filename, 'r') as f:
		content = f.read()
		key = filename[0:-4]
		if tags[key] != "idk/skip":
			trainMetadata.append({"text":content, "label":tags[key], "user":None})

#########################################################

client = MongoClient('mongodb+srv://haspburn71280:H8IsNoGood@hatebaddb-kbv0e.gcp.mongodb.net/test?retryWrites=true&w=majority')
db = client.hatebad
test = db.hateTestSet
train = db.hateTrainingSet

test.insert_many(testMetadata)
train.insert_many(trainMetadata)