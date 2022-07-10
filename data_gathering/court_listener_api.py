import json
import requests

URL = "https://www.courtlistener.com/api/rest/v3/opinions/?court=scotus&judge=sotomayor"

response = requests.get(URL)
data = response.json()
#print(data)

with open('sotomayor_opinions.json', 'w') as f:
    json.dump(data, f)