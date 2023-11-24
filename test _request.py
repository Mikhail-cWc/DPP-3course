import requests
import plotly.graph_objects as go
import plotly
import os
import json
# Указываем адрес сервера Flask API
url = 'http://127.0.0.1:5000/predict'

# Загружаем файлы
files = {
    'file1': ('100.dat', open('mit-database/100.dat', 'rb')),
    'file2': ('100.hea', open('mit-database/100.hea', 'rb')),
    'file3': ('100.atr', open('mit-database/100.atr', 'rb')),
}

# Отправляем POST запрос с файлами
response = requests.post(url, files=files)

# Получаем результат
result = response.json()

fig = plotly.io.from_json(json.dumps(result['html']), output_type="FigureWidget", skip_invalid=False, engine = None)

fig.show()

print('GPT analysis', result['text'])
