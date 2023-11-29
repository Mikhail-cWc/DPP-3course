from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pipline import BasePipeline, EcgPipelineDataset1D
from model.model import ECGnet
import torch
import json
from openai import OpenAI
from dotenv import load_dotenv
import base64

load_dotenv()
client = OpenAI()
app = Flask(__name__)

model = ECGnet()
model.load_state_dict(torch.load('./model/model.pth', map_location='cpu'))

UPLOAD_FOLDER = 'uploads'
JSON_STORE = "html"
IMAGES_STORE = "images"


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(IMAGES_STORE):
    os.makedirs(IMAGES_STORE)
if not os.path.exists(JSON_STORE):
    os.makedirs(JSON_STORE)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_FOLDER'] = JSON_STORE
app.config['IMAGES_FOLDER'] = IMAGES_STORE
ALLOWED_EXTENSIONS = {'dat', 'hea', 'atr'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

def request_chatgpt(image_path):
    base64_image = encode_image(image_path)
    
    completion = client.chat.completions.create(
        model = "gpt-4-vision-preview",
        messages = [
            {
                "role": "system", 
                 "content": "Привет, я - MedAI, специализированный на анализе ЭКГ. \
             Пожалуйста, отправьте мне обработанный график ЭКГ с метками вашей AI-модели \
             (например, стрелки и буквы, где пустота означает 'Норма'). Моя задача - предоставить \
             анализ и рекомендации по этому графику [хотя бы примерный]. Я сосредоточусь на \
             выявлении потенциальных проблем и дискомфорта, связанных с этим графиком, \
             исключительно на основе представленных данных, без упоминания о необходимости \
             дополнительной информации. На графике по оси X каждые 8 миллисекунд соответствуют\
              одной временной единице. Все ответы будут предоставлены строго на русском языке."
            },
				{
               "role": "user", 
     			 "content": 
                 [
                    {
                        "type":"text",
                        "text":"Опиши фотографию."
						  },
                    {
							 	"type":"image_url",
                        "image_url": 
                           {
                            "url":f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
									}
						  }
                  ]
				}
		  ]
	 )
    print(completion)
    return completion['choices'][0]['message']['content']
    

@app.route('/predict', methods=['POST'])
def predict():
    try:
		
        if 'file1' not in request.files or 'file2' not in request.files or 'file3' not in request.files:
            return jsonify({'error': 'No files provided'})

        file1 = request.files['file1']
        file2 = request.files['file2']
        file3 = request.files['file3']

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
        
        figure_content = json.load(open(f"./{app.config['JSON_FOLDER']}/{filename1[:-4]}.json"))
        #text_content = request_chatgpt(f"./{app.config['IMAGES_FOLDER']}/{filename1[:-4]}.jpeg")
        text_content = "adfasdf"
        os.remove(file1_path)
        os.remove(file2_path)
        os.remove(file3_path)
        os.remove(f"./{app.config['JSON_FOLDER']}/{filename1[:-4]}.json")
        os.remove(f"./{app.config['IMAGES_FOLDER']}/{filename1[:-4]}.jpeg")
        return jsonify({'ecg': figure_content, 'text': text_content})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
