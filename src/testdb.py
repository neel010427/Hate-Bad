from pymongo import MongoClient

client = MongoClient('mongodb+srv://haspburn71280:H8IsNoGood@hatebaddb-kbv0e.gcp.mongodb.net/test?retryWrites=true&w=majority')
db = client.hatebad #database

a = db.positiveTestSet.aggregate(
   [ { $sample: { size: 3 } } ]
)

print(a)