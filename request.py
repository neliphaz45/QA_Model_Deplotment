import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'question':"From which country is the band Queen ?"})

print(r.json())