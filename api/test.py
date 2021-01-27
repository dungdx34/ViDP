'''
Dung Doan
'''
import requests

url_postag = 'http://localhost:8880/nlp/parser'
input_file = 'vnexpress.txt'

with open(input_file, 'r') as in_file:
    lines = in_file.readlines()
    for line in lines:
        line = line.strip()

        data = {
            "sentence": line
        }
        try:
            print(line)
            response = requests.post(url_postag, json=data)
            if response.status_code == 200:
                json = response.json()
                result = json['result']
                print(result)

            else:
                print("Request Error, Code: {0}".format(response.status_code))
        except requests.exceptions.RequestException:
            pass

