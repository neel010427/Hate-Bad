from pymongo import MongoClient

client = MongoClient('mongodb+srv://haspburn71280:H8IsNoGood@hatebaddb-kbv0e.gcp.mongodb.net/test?retryWrites=true&w=majority')
db = client.hatebad
test = db.hateTestSet

#testDoc = {"First Name":"Kyle", "Last Name":"Zhou", "Age":20, "Urmum":"gay"}
#test.insert_one(testDoc)

x = test.find({})

for i in x:
	print(type(i['text']))
