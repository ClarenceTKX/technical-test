import requests
import pandas as pd
from pathlib import Path
import configparser
import os

# setup parameters with configuration file (no need to interact with cv-decode.py)
config = configparser.ConfigParser()
config.read(r'asr/cv-decode.cfg')

# setup API link (ensure that asr_api is up)
api_url = 'http://localhost:8001/asr'

# store the paths from the configuration file
base_file_path = Path(config.get('Paths', 'AudioFilesDirectory'))
csv_path = config.get('Paths', 'CSVFilePath')

# Read the CSV file, and add a new column for generated text
df = pd.read_csv(csv_path)
df['generated_text'] = ''

# Function to transcribe an audio file
def transcribe_audio(file_path):

    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(api_url, files=files)

    if response.status_code == 200:
        return response.json().get('transcription', 'No transcription available')
    else:
        return f'Error in transcription: {response.content}'

for index, row in df.iterrows():
    audio_file_full = base_file_path / row['filename']
    if audio_file_full.is_file():
        try:
            transcription = transcribe_audio(audio_file_full)
            df.at[index, 'generated_text'] = transcription
        except Exception as e:
            print(f"Error processing file {audio_file_full}: {e}")

# Save the updated DataFrame
df.to_csv('asr/updated_cv-valid-dev.csv', index=False)
