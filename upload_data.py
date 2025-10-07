from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri="mongodb+srv://aashu221301_db_user:U21MVdJkaqXvO6Dn@cluster0.2ovymbd.mongodb.net/?retryWrites=true&w=majority"

#create a new client and connect to server
client=MongoClient(uri)

#create database name and collection name
DATABASE_NAME="ASkills"
COLLECTION_NAME="waferfault"

df=pd.read_csv("C:\Users\aashu\OneDrive\Desktop\sensor_project\notebooks\wafer_23012020_041211 (1).csv")

df=df.drop("Unnamed: 0",axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)