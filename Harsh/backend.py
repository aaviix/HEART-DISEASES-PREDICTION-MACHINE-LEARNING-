from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import random

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/get-options', methods=['GET'])
def get_options():
    i = random.choice([True, False])
    if i:
        return jsonify(message='I am dumb as fuck')
    return jsonify(message='No, I am not dumb. Shut up')

@app.route('/multiply', methods=['POST'])
def multiply():
    data = request.json
    number = data.get('number', 1)
    result = number * 2
    return jsonify(result=result)

if __name__ == '__main__':
    socketio.run(app, port=8080, debug=True)
