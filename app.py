import nest_asyncio
nest_asyncio.apply()  # Patch asyncio to allow nested event loops

import streamlit as st
import torch
import torchaudio
from transformers import pipeline

import os
import tempfile
from datetime import datetime

torchaudio.set_audio_backend("soundfile")

def transcribe_audio_file(filename, chunk_duration=29):
    """
    Splits the input WAV file into chunks of 'chunk_duration' seconds,
    transcribes each chunk using the asr_pipeline, and returns the combined text.
    
    Parameters:
        filename (str): Path to the long audio file.
        chunk_duration (int): Duration of each chunk in seconds. Default is 30 seconds.
    
    Returns:
        str: The combined transcription of all audio chunks.
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(filename)
    chunk_samples = sample_rate * chunk_duration
    full_text = ""
    
    num_samples = waveform.shape[1]
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {waveform.shape}')

    for start in range(0, num_samples, chunk_samples):
        # Define the chunk indices
        end = min(start + chunk_samples, num_samples)
        chunk_waveform = waveform[:, start:end]
        
        # Write the current chunk to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
        torchaudio.save(temp_filename, chunk_waveform, sample_rate)
        
        # Transcribe the temporary audio file using the ASR pipeline
        result = asr_pipeline(temp_filename)
        # print('chunk processed')
        full_text += result.get('text', '') + " "
        
        # Remove the temporary file
        os.remove(temp_filename)
        
    return full_text.strip()

# Example usage:
# transcription = transcribe_audio_file("long_audio.wav")
# print(transcription)

# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def load_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

asr_pipeline = load_model()

def main():
    
    st.title("Varsha's Audio Transcription App")
    st.write("Upload an audio file to get its transcription.")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Save the uploaded file to a temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner('Transcribing...'):
            print(f'Transcription started : {datetime.now()}')
            result = transcribe_audio_file("temp_audio.wav")
            print(f'Transcript generated : {datetime.now()}')
    
        # st.subheader("Here is your transcription: \n")
        st.write(result)

if __name__ == "__main__":
    main()


