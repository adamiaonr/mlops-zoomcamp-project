from datetime import datetime

import requests

time_of_day = datetime.now()
time_of_day_seconds = (
    time_of_day - time_of_day.replace(hour=0, minute=0, second=0, microsecond=0)
).total_seconds()

stop_info = {
    "PublishedLineName": "S40",
    "DirectionRef": 0,
    "NextStopPointName": "SOUTH AV/ARLINGTON PL",
    "TimeOfDayInSeconds": int(time_of_day_seconds),
    "DayOfWeek": 4,
}

print(f"QUERY: {stop_info}")

url = 'http://bus-delay-prediction.local/predict'
response = requests.post(url, json=stop_info)
print(f"RESPONSE: {response.json()}")
