from flask import Flask, render_template, request, jsonify
import whisper
import os
import tempfile

app = Flask(__name__)

model = whisper.load_model("base")

def transcribe_audio(file_path):
    result = model.transcribe(file_path, task="transcribe")
    return result.get('text', 'Transcription failed'), result.get('language', 'Unknown')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files or request.files['audio'].filename == '':
        return jsonify({"error": "No audio file uploaded or selected"}), 400

    file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_path = temp_audio.name
        file.save(temp_path)

    try:
        transcription, language = transcribe_audio(temp_path)
        return jsonify({"transcription": transcription, "language": language})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True)
