from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForCTC
import soundfile as sf
from pydub import AudioSegment
import io
import torch 

app = Flask(__name__)

#task 2b: ping API to return 'pong' 
@app.route('/ping', methods=['GET'])
def ping_pong():
    return 'pong'

#load the Wav2Vec2 Model (Large)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Load model directly (for fine-tuning task)
# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h")
# model = AutoModelForCTC.from_pretrained("ClarenceTKX/checkpoints")

#task 2c: write an API for the wav2vec2 (large) model
@app.route('/asr', methods=["POST"])
def asr():
    #exception handling for file var
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    #handles audio processing and transformation
    '''
    Parameters:
    bitrate = 16000 (specified by wav2vec2 model)
    format = *.wav
    '''
    audio = AudioSegment.from_file(file, format=file.filename.split('.')[-1])
    audio = audio.set_frame_rate(16000).set_channels(1)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)

    #decoding the file
    waveform, _ = sf.read(buffer)
    buffer.close()

    # Process with wav2vec2 (code provided by HF model card)
    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # Calculate the duration (D = length of waveform / specified sample rate)
    duration = len(waveform) / 16000.0

    # return results of transcription in JSON format as specified
    return jsonify({'transcription': transcription, 'duration': f"{duration:.1f}"})


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8001)