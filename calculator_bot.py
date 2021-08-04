import requests
import json
from flask import Flask, request, jsonify


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/', methods=['POST', 'GET'])
def webhook():
    req = request.get_json(silent=True, force=True)
    sum = 0
    query_result = req.get('queryResult')
    print(query_result)
    num1 = int(query_result.get('parameters').get('number'))
    num2 = int(query_result.get('parameters').get('number1'))
    sum = str(num1 + num2)
    print('here num1 = {0}'.format(num1))
    print('here num2 = {0}'.format(num2))
    return {
        "fulfillmentText": 'The sum of the two numbers is: '+sum,
        "displayText": '25',
        "source": "webhookdata"
    }




if __name__ == '__main__':
    app.run(host='0.0.0.0')    