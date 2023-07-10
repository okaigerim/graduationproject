import requests

adress = "Алматы сатпаева 22"
url = 'https://geocode-maps.yandex.ru/1.x/?apikey=f3fe04e3-2e96-4c00-acd8-6d8a1d0825d2&format=json&geocode=' + adress
req = requests.get(url)
response = req.json()
print(response)
