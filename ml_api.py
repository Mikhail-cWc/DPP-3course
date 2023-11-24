from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pipline import BasePipeline, EcgPipelineDataset1D
from model.model import ECGnet
import torch
import json

app = Flask(__name__)

model = ECGnet()
model.load_state_dict(torch.load('./model/model.pth', map_location='cpu'))

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
JSON_STORE = "html"
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists(JSON_STORE):
    os.makedirs(JSON_STORE)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'dat', 'hea', 'atr'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def request_chatgpt():
    pass

@app.route('/predict', methods=['POST'])
def predict():
    try:
		
        if 'file1' not in request.files or 'file2' not in request.files or 'file3' not in request.files:
            return jsonify({'error': 'No files provided'})

        file1 = request.files['file1']
        file2 = request.files['file2']
        file3 = request.files['file3']

        # Проверяем расширения файлов
        if not allowed_file(file1.filename) or not allowed_file(file2.filename) or not allowed_file(file3.filename):
            return jsonify({'error': 'Invalid file extension'})

        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filename3 = secure_filename(file3.filename)

        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file3_path = os.path.join(app.config['UPLOAD_FOLDER'], filename3)

        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        file3.save(os.path.join(app.config['UPLOAD_FOLDER'], filename3))
        
        data_loader = EcgPipelineDataset1D(file1_path[:-4])
                              
        pipline = BasePipeline(model, data_loader, filename1[:-4], beats = 5)

        pipline.run_pipeline()
        
        figure_content = json.load(open(f"./{JSON_STORE}/{filename1[:-4]}.json"))
        
        os.remove(file1_path)
        os.remove(file2_path)
        os.remove(file3_path)
        return jsonify({'html': figure_content, 'text': "КАКО-ТО ТЕКСТ ГПТ"})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
