import json
import urllib.request


request = urllib.request.Request(
    "http://127.0.0.1:8765/retrieve",
    data=json.dumps({"query": "How do I rebuild pgvector?"}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    for item in json.loads(response.read()):
        print(item["name"], "-", item["description"])