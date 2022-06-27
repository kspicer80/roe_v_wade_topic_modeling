import json
import requests

URL = "https://www.courtlistener.com/api/rest/v3/opinions/?cluster__docket__court__id=scotus&court__date_modified__gt=2020-01-01T00:00:00Z"

response = requests.get(URL)
data = response.json()
print(data)

with open('2020_to_today_opinions.json', 'w') as f:
    json.dump(data, f)