import requests
files = {'file': open('strings4.wav', 'rb')}
requests.post('http://192.168.1.36:5000/file-upload',files=files)
