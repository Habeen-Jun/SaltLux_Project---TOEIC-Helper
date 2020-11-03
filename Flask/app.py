import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from ToeicAudioProcessor import ToeicAudioProcessor
from datetime import datetime
from ResponseVO import Object
from hparams import Question_Segmentation
from utils import GetCurrentDatetime
from mainprocess import process_toeic_full_test, process_toeic_full_test_Arraylist

UPLOAD_DIR = os.path.abspath(os.getcwd())

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR

@app.route('/')
def upload_main():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>File Upload Toeic Audio File</title>
    </head>
    <body>
        <form action="http://49.50.167.72:8000/file-upload" method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit">
        </form>
    </body>
    </html>"""

@app.route('/file-upload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'GET':
        path = request.args['uploaded_file_path']
        response = process_toeic_full_test(path)
        return response
    elif request.method == 'POST':
        f = request.files['file']
        fname = secure_filename(f.filename)
        path = os.path.join(os.getcwd(), fname)
        f.save(path)
        response = process_toeic_full_test_Arraylist(path)
        # response = process_toeic_full_test(path)

        # with open('res_example.json', 'w') as f:
        #     f.write(response)

        return response.toJSON()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)