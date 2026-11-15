from pymongo import AsyncMongoClient

from core.settings import settings

# Lazy: no socket is opened until the first operation.
client = AsyncMongoClient(settings.mongo_uri)
db = client.get_default_database()
