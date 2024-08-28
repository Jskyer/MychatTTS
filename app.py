from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import model_core

app = Flask(__name__)
# 跨域配置
CORS(app)

tts_manager = model_core.TTSManager()

FILE_OUTPUT_PREFIX = "C:/devFiles/projects/chatTTS/output/"

@app.route('/health', methods=['POST'])
def checkHealth():
    print("health")
    return jsonify({'code': 200, 'success': True, 'data': 'app is running'})

@app.route('/tts/speech', methods=['POST'])
def getTTSResp():
    data = request.json
    texts = data.get('speech')
    if not texts:
        return jsonify({"error": "缺少必需参数speech"}), 400

    resp = tts_manager.generate_audio(texts)
    # refined_text = resp[1]
    return send_file(FILE_OUTPUT_PREFIX + resp)
    # return jsonify({'code': 200, 'success': True, 'data': resp})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)