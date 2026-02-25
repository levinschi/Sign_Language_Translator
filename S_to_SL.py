import tkinter as tk
from tkinter import scrolledtext
import pyaudio
import wave
import threading
import speech_recognition as sr
from PIL import Image, ImageTk

# Dictionary mapping letters to sign language images
letter_to_image = {
    chr(i): f"hand1_{chr(i)}_bot_seg_1_cropped.jpg" for i in range(97, 123)  # a-z
}

# Audio Recording and Transcription Section
is_recording = False
frames = []
audio_thread = None

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILE = "output.wav"


def record_audio():
    global is_recording, frames
    is_recording = True
    frames = []
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


def start_recording():
    global audio_thread, is_recording
    translator_app.image_label.config(image="")
    translator_app.image_label.image = None
    if not is_recording:
        status_label.config(text="Status: Recording...")
        audio_thread = threading.Thread(target=record_audio)
        audio_thread.start()


def stop_recording():
    global is_recording
    is_recording = False
    if audio_thread is not None:
        audio_thread.join()
    status_label.config(text="Status: Processing audio...")
    transcribe_audio()


def transcribe_audio():
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(OUTPUT_FILE) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            text_display.delete(1.0, tk.END)
            text_display.insert(tk.END, text)
            translator_app.load_text_from_display()  # Automatically load the text for translation
            status_label.config(text="Status: Transcription complete!")
    except Exception as e:
        text_display.delete(1.0, tk.END)
        text_display.insert(tk.END, f"Error: {str(e)}")
        status_label.config(text="Status: Error during transcription")


# Sign Language Translator Section
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        
        # Variables for input text and current letter position
        self.input_text = ""
        self.current_index = 0
        
        self.image_label = tk.Label(root)
        self.image_label.place(x=850, y=50)

        # Resize the button to match the others
        self.next_button = tk.Button(root, text="Next Letter", command=self.show_next_letter, bg="blue", fg="white", width=20, height=2)
        self.next_button.grid(pady=10)
        self.next_button.place(x=50, y=150)

    def load_text_from_display(self):
        self.input_text = text_display.get("1.0", tk.END).strip().lower()
        self.current_index = 0 
        status_label.config(text="Status: Ready to display letters")

    def show_next_letter(self):
        if not self.input_text:
            self.load_text_from_display()

        if self.current_index >= len(self.input_text):
            self.current_index = 0
            self.image_label.config(image="")
            text_display.tag_remove("processed", "1.0", tk.END)
            text_display.tag_remove("current", "1.0", tk.END)
            text_display.tag_remove("remaining", "1.0", tk.END)
            status_label.config(text="Status: Translation complete!")
            return
        
        current_letter = self.input_text[self.current_index]
        if current_letter in letter_to_image:
            image_path = letter_to_image[current_letter]
            try:
                img = Image.open(image_path)
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo)
                self.image_label.image = photo
            except Exception as e:
                print(f"Error loading image for {current_letter}: {e}")
        else:
            self.image_label.config(image="")
        
        # Highlight the text in the text_display
        self.update_highlight()

        # Move to the next letter
        self.current_index += 1

    def update_highlight(self):
        # Reset highlight
        text_display.tag_remove("processed", "1.0", tk.END)
        text_display.tag_remove("current", "1.0", tk.END)
        text_display.tag_remove("remaining", "1.0", tk.END)
        
        # Highlight processed letters
        for i in range(self.current_index):
            text_display.tag_add("processed", f"1.{i}", f"1.{i+1}")
        
        # Highlight the current letter
        if self.current_index < len(self.input_text):
            text_display.tag_add("current", f"1.{self.current_index}", f"1.{self.current_index+1}")
        
        # Highlight remaining letters
        for i in range(self.current_index + 1, len(self.input_text)):
            text_display.tag_add("remaining", f"1.{i}", f"1.{i+1}")


# Main Window and Layout
root = tk.Tk()
root.geometry("1100x350")

# Status label with padding applied only to the text label in row 0
status_label = tk.Label(root, text="Status: Idle", font=("Arial", 14))
status_label.grid(columnspan=2, pady=10, padx=50)
status_label.place(x=40, y=10)

# Start and Stop buttons for recording
start_button = tk.Button(root, text="Start Recording", command=start_recording, width=20, height=2, bg="green", fg="white")
start_button.grid(padx=10, pady=5)
start_button.place(x=50, y=50)

stop_button = tk.Button(root, text="Stop Recording", command=stop_recording, width=20, height=2, bg="red", fg="white")
stop_button.grid(row=2, column=1, padx=10, pady=5)
stop_button.place(x=50, y=100)

# Text display for transcription (right side)
text_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12), width=60, height=10)
text_display.grid(rowspan=2, padx=10, pady=10)
text_display.place(x=300, y=50)

# Highlight tag configuration
text_display.tag_configure("processed", foreground="gray")
text_display.tag_configure("current", foreground="blue")
text_display.tag_configure("remaining", foreground="black")

# Initialize the Sign Language Translator
translator_app = SignLanguageApp(root)

root.mainloop()
