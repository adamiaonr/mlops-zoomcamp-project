import requests

stop_info = {
    "PublishedLineName": "S40",
    "DirectionRef": 0,
    "NextStopPointName": "SOUTH AV/ARLINGTON PL",
    "TimeOfDayInSeconds": 10400,
    "DayOfWeek": 4,
}

url = 'http://bus-delay-prediction.local/predict'
response = requests.post(url, json=stop_info)
print(response.json())
