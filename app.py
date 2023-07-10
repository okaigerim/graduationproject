import joblib
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import math
from flask import Flask, request, jsonify, render_template
import pickle
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
model = open('rf_model.pkl', 'rb')
rf_model = joblib.load(model)


def get_azimuth(latitude, longitude):
    rad = 6372795
    city_center_coordinates = [43.238293, 76.945465]
    llat1 = city_center_coordinates[0]
    llong1 = city_center_coordinates[1]
    llat2 = float(latitude)
    llong2 = float(longitude)

    lat1 = llat1 * math.pi / 180.
    lat2 = llat2 * math.pi / 180.
    long1 = llong1 * math.pi / 180.
    long2 = llong2 * math.pi / 180.

    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)

    y = math.sqrt(math.pow(cl2 * sdelta, 2) + math.pow(cl1 * sl2 - sl1 * cl2 * cdelta, 2))
    x = sl1 * sl2 + cl1 * cl2 * cdelta
    ad = math.atan2(y, x)

    x = (cl1 * sl2) - (sl1 * cl2 * cdelta)
    y = sdelta * cl2
    z = math.degrees(math.atan(-y / x))

    if x < 0:
        z = z + 180.

    z2 = (z + 180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2 * math.pi) * math.floor((z2 / (2 * math.pi))))
    angledeg = (anglerad2 * 180.) / math.pi

    return round(angledeg, 2)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # print(int_features)
    wallmaterial = request.form.get('wallmaterial')
    floorNumber = request.form.get('floorNumber')
    floorsTotal = request.form.get('floorsTotal')
    state = request.form.get('state')
    totalArea = request.form.get('totalArea')
    year = request.form.get('year')
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')
    flat = pd.DataFrame({
        'wallmaterial': [wallmaterial],
        'floorNumber': [floorNumber],
        'floorsTotal': [floorsTotal],
        'state': [state],
        'totalArea': [totalArea],
        'year': [year],
        'latitude': [latitude],
        'longitude': [longitude]
    })
    city_center_coordinates = [43.238293, 76.945465]
    flat['distance'] = list(
        map(lambda x, y: geodesic(city_center_coordinates, [x, y]).meters, flat['latitude'], flat['longitude']))
    flat['azimuth'] = list(map(lambda x, y: get_azimuth(x, y), flat['latitude'], flat['longitude']))
    flat['distance'] = flat['distance'].astype(float)
    flat['azimuth'] = flat['azimuth'].astype(float)
    flat['distance'] = flat['distance'].round(0)
    flat['azimuth'] = flat['azimuth'].round(0)
    flat = flat.drop('latitude', axis=1)
    flat = flat.drop('longitude', axis=1)
    prediction = rf_model.predict(flat).round(0)
    flat['totalArea'] = flat['totalArea'].astype(float)
    price = prediction * flat['totalArea']
    # print(f'Apartment price predicted with the model: {int(price[0].round(-3))} KZT')
    output = int(price[0].round(-3))
    return render_template('index.html', prediction_text='Approximate price of an apartment: {}'.format(output))


@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
