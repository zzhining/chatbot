import seq2seq
import requests
import json
from flask import Flask, request, jsonify


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/', methods=['POST', 'GET'])
def webhook():
    req = request.get_json(silent=True, force=True)
    query_result = req.get('queryResult')
    intent = query_result.get('intent').get('displayName')
    print(intent)
    
    if intent == 'Default Fallback Intent':
    #if intent == 'Default Welcome Intent':
        text = query_result.get('queryText')
        answer = seq2seq.get_answer(text)
    else:
        answer = 'error'
        
    res = {'fulfillmentText': answer}
        
    return jsonify(res)


if __name__ == '__main__':    
    # Seq2Seq 모델 로드
    seq2seq.load()
    app.run(host='0.0.0.0')    