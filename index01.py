import requests
import json
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/', methods=['POST', 'GET'])
def webhook():    
    return {
       "fulfillmentText": 'This is from the replit webhook',
       "source": 'webhook'
    }    
    
@app.route('/test')
def test(): 
    return "<h1>This is test page!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0')    