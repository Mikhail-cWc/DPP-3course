import requests
import plotly
import json

# Указываем адрес сервера Flask API
url = 'http://127.0.0.1:5000/predict'

# Загружаем файлы
files = {
    'file1': ('101.dat', open('mit-database/101.dat', 'rb')),
    'file2': ('101.hea', open('mit-database/101.hea', 'rb')),
    'file3': ('101.atr', open('mit-database/101.atr', 'rb')),
}

# Отправляем POST запрос с файлами
response = requests.post(url, files=files)

# Получаем результат
result = response.json()

print(result["text"])
fig = plotly.io.from_json(json.dumps(result['ecg']), output_type="FigureWidget", skip_invalid=False, engine = None)

fig.show()
