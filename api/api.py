import requests
import time

# API URL
url = "https://europe-west1-friendly-medley-412321.cloudfunctions.net/calculate-price"

# JSON data to be sent to the API
data = {
    "m": 50,
    "r": 0.1,
    "T": 0.2,
    "sigma": 0.3,
    "S0": 100,
    "K": 100,
    "H": 85,
    "q": 0,
    "confidence_level": 0.95,
    "n_paths": 1*10**7
}

start = time.time()

# Making the POST request
response = requests.post(url, json=data)

# Checking if the request was successful
if response.status_code == 200:
    print("Response from API: ", response.json())
else:
    print("Error in API call: ", response.status_code, response.text)

end = time.time()
print(round(end-start,1), "Seconds")
