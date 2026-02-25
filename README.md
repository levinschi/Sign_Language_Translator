# Sign Language Translator

A desktop app for converting speech to sign language images and recognizing sign language to text using a webcam.

## Overview
- Speech to Sign Language: records audio, transcribes it with Google Speech Recognition, and shows sign images for each letter.
- Sign Language to Text: uses a trained model with MediaPipe hand landmarks to classify hand signs from the webcam.

## Requirements
- Python 3.9+
- Windows (tested)
- Microphone and webcam

## Python dependencies
Install with pip:
```
customtkinter
mediapipe
numpy
opencv-python
pillow
pyaudio
speechrecognition
```

## Run
1. Open a terminal in this folder.
2. Install dependencies.
3. Run:
```
python "proiect final.py"
```

## Project files
- proiect final.py: main GUI app
- clasificator.py: webcam sign recognition demo
- S_to_SL.py: speech to sign demo
- model.p: trained classifier model
- hand1_*_cropped.jpg: sign images
- assets: PNG/JPG UI backgrounds

## Notes
- The speech feature uses the Google Speech Recognition API through the `speechrecognition` library.
- If `pyaudio` fails to install, install the PortAudio build for Windows first.
- `output.wav` is generated at runtime and ignored by git.
 - The sign-letter images used for speech-to-sign output are not included. They were sourced from the internet, and I cannot redistribute them due to copyright.
