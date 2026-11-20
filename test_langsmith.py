

import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(override=True)

api_key = os.getenv("LANGSMITH_API_KEY")

if not api_key:
    raise ValueError("LANGSMITH_API_KEY not found!")

print("API key detected:", bool(api_key))

client = Client(api_key=api_key)

datasets = list(client.list_datasets())

print("Connected successfully!")

for dataset in datasets:
    print(dataset.name)
