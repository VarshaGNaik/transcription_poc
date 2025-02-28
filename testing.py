import streamlit as st
import torch
import torchaudio
from transformers import pipeline

def load_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

asr_pipeline = load_model()

def run_model(filename: str):
    result = asr_pipeline(filename)

    return result['text']

def main():
    fn = input("Please enter full filename")

    with open('output.txt', 'a') as f:
        f.write(f'Filename: {fn} \n')
        f.write(run_model(fn))



if __name__ == '__main__':
    main()