import tkinter as tk
from tkinter import *
from tkinter import scrolledtext
import pyaudio
import wave
import threading
import speech_recognition as sr
from PIL import Image, ImageTk
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from customtkinter import *
from tkinter import font

colour1 = '#020f12'
colour2 = '#05d7ff'
colour3 = '#65e7ff'
colour4 = 'BLACK'


def create_main_menu():
    
    root = tk.Tk()
    root.title("Sign Language Translator")

    root.geometry("800x600")

    bg_image = Image.open("fundal_men.png")
    bg_photo = CTkImage(light_image=bg_image, size=(800, 600))
    background_label = CTkLabel(root, image=bg_photo, text="")
    background_label.image = bg_photo
    background_label.place(relwidth=1, relheight=1)

    def exit_fullscreen(event=None):
        root.attributes('-fullscreen', False)
    root.bind("<Escape>", exit_fullscreen)

    def open_speech_to_sign_language():
        root.destroy()  
        open_window_speech_language("Speech to Sign Language")

    def open_sign_language_to_text():
        root.destroy() 
        open_window_language_text()

    button_text_to_sign = CTkButton(
        master=root,
        fg_color="#00171F", 
        hover_color="#00A7E1", 
        text_color="#FFFFFF",
        font=("Arial", 20),
        text="Speech to Sign Language",
        corner_radius = 32,
        width= 100,
        height= 80, 
        command=open_speech_to_sign_language
    )
    button_text_to_sign.pack(pady=20)
    button_text_to_sign.place(relx=0.5, rely=0.4, anchor="center")

    button_sign_to_text = CTkButton(
        master=root,
        fg_color="#00171F", 
        hover_color="#00A7E1", 
        text_color="#FFFFFF",
        font=("Arial", 20),
        corner_radius = 32,
        width= 100,
        height= 80,
        text="  Sign Language to Text  ", 
        command=open_sign_language_to_text
    )
    button_sign_to_text.pack(pady=20)
    
    button_sign_to_text.place(relx=0.5, rely=0.6, anchor="center")  
    root.mainloop()

def open_window_speech_language(title):
    letter_to_image = {
        chr(i): f"hand1_{chr(i)}_bot_seg_1_cropped.jpg" for i in range(97, 123)  
    }

    global is_recording 
    is_recording= False
    frames = []
    audio_thread = None

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    OUTPUT_FILE = "output.wav"


    def set_status(text):
        root.after(0, lambda: status_label.config(text=text))

    def record_audio():
        global is_recording, frames
        is_recording = True
        frames = []
        audio = None
        stream = None

        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

            with wave.open(OUTPUT_FILE, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)

                while is_recording:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    wf.writeframes(data)
        except Exception as exc:
            set_status(f"Status: Recording error: {exc}")
            is_recording = False
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if audio is not None:
                audio.terminate()


    def start_recording():
        global audio_thread, is_recording
        translator_app.image_label.config(image="")
        translator_app.image_label.image = None
        if not is_recording:
            status_label.config(text="Status: Recording...")
            audio_thread = threading.Thread(target=record_audio, daemon=True)
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
                translator_app.load_text_from_display()
                status_label.config(text="Status: Transcription complete!")
        except Exception as e:
            text_display.delete(1.0, tk.END)
            text_display.insert(tk.END, f"Error: {str(e)}")
            status_label.config(text="Status: Error during transcription")

    class SignLanguageApp:
        def __init__(self, new_window):
            self.root = new_window
            self.root.title("Sign Language Translator")
            
            self.input_text = ""
            self.current_index = 0
            
            self.image_label = tk.Label(root)
            self.image_label.place(x=858, y=50)

            self.next_button = CTkButton(
                master=root, 
                text="    Next Letter    ", 
                command=self.show_next_letter, 
                corner_radius = 32, 
                fg_color="#00171F", 
                hover_color="#00A7E1", 
                text_color="#FFFFFF",
                font=("Arial", 20), 
                border_width=2, 
            
                )
            self.next_button.grid(pady=10)
            self.next_button.place(x=40, y=160)

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
            
            self.update_highlight()

            self.current_index += 1

        def update_highlight(self):
            text_display.tag_remove("processed", "1.0", tk.END)
            text_display.tag_remove("current", "1.0", tk.END)
            text_display.tag_remove("remaining", "1.0", tk.END)
            
            for i in range(self.current_index):
                text_display.tag_add("processed", f"1.{i}", f"1.{i+1}")
            
            if self.current_index < len(self.input_text):
                text_display.tag_add("current", f"1.{self.current_index}", f"1.{self.current_index+1}")
            
            for i in range(self.current_index + 1, len(self.input_text)):
                text_display.tag_add("remaining", f"1.{i}", f"1.{i+1}")

    def back_to_main_menu():
        root.destroy()
        create_main_menu() 
        
    root = tk.Tk()
    root.geometry("1100x300")

    bg_image = Image.open("speech_to_SL.png")
    bg_photo = CTkImage(light_image=bg_image, size=(1100, 300))
    background_label = CTkLabel(root, image=bg_photo, text="")
    background_label.image = bg_photo
    background_label.place(relwidth=1, relheight=1)

    status_label = tk.Label(root, text="Status: Idle", font=("Arial", 14))
    status_label.grid(columnspan=2, pady=10, padx=50)
    status_label.place(x=40, y=20)

    start_button = CTkButton(
        master=root, 
        text="Start Recording", 
        command=start_recording, 
        corner_radius = 32, 
        fg_color="#00D100", 
        hover_color="#00A7E1", 
        text_color="#FFFFFF",
        font=("Arial", 20), 
        border_width=2, 
        
        )
    start_button.grid(padx=10, pady=5)
    start_button.place(x=40, y = 60)

    stop_button = CTkButton(
        master=root, 
        text="Stop  Recording", 
        command=stop_recording, 
        corner_radius = 32, 
        fg_color="#FF0000", 
        hover_color="#00A7E1", 
        text_color="#FFFFFF",
        font=("Arial", 20), 
        border_width=2 

        )
    stop_button.grid(padx=10, pady=5)
    stop_button.place(x=40, y=110)
    
    button_back = CTkButton(
        master=root, 
        text="        Back        ",
        command=back_to_main_menu,
        corner_radius = 32, 
        fg_color="#003459", 
        hover_color="#00A7E1", 
        text_color="#FFFFFF",
        font=("Arial", 20), 
        border_width=2
    )
    button_back.place(x=40, y=220)
    bold_font = font.Font(family="Arial", size=12, weight="bold")
    text_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=bold_font, width=60, height=10)
    text_display.grid(padx=10, pady=10)
    text_display.place(x=250, y=60)

    text_display.tag_configure("processed", foreground="gray")
    text_display.tag_configure("current", foreground="blue")
    text_display.tag_configure("remaining", foreground="black")

    translator_app = SignLanguageApp(root)

    def on_close():
        global is_recording
        is_recording = False
        if audio_thread is not None and audio_thread.is_alive():
            audio_thread.join(timeout=1)
        try:
            root.quit()
        except tk.TclError:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()

def open_window_language_text():
    new_window = CTk()
    new_window.title("Sign Language to text")

    new_window.geometry("950x480") 

    bg_image = Image.open("SL_to_text.png")

    bg_photo = CTkImage(light_image=bg_image, size=(950, 480))
    background_label = CTkLabel(new_window, image=bg_photo, text="")
    background_label.image = bg_photo
    background_label.place(relwidth=1, relheight=1)

    video_label = tk.Label(new_window)
    video_label.place(x=540, y=100, width=640, height=480) 

    cap = None
    update_job = None
    is_running = False

    def stop_SL():
        nonlocal cap, update_job, is_running
        is_running = False
        if update_job is not None:
            try:
                video_label.after_cancel(update_job)
            except tk.TclError:
                pass
            update_job = None
        if cap is not None:
            cap.release()
            cap = None

    def start_SL():
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']

        nonlocal cap, update_job, is_running
        stop_SL()
        cap = cv2.VideoCapture(0)
        is_running = True

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
            8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
            16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
            24: 'Y', 25: 'Z', 26: 'Space', 27: 'Delete'
        }

        last_predicted_char = None
        prediction_time = None

        def update_frame():
            nonlocal last_predicted_char, prediction_time, update_job
            if not is_running:
                return

            ret, frame = cap.read()
            if not ret:
                return

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) > 1:
                    cv2.putText(frame, "Error: More than one hand detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
                    hand_landmarks = results.multi_hand_landmarks[0]
                else:
                    hand_landmarks = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_, y_ = [], []
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                current_text = text_area.get(1.0, tk.END).strip()

                if predicted_character == last_predicted_char:
                    if prediction_time is None:
                        prediction_time = time.time()
                    elif time.time() - prediction_time > 1.25:
                        
                        if predicted_character == 'Space':
                            text_area.insert(tk.END, f"{' '}")
                        
                        elif predicted_character == 'Delete':
                            new_text = current_text[:-1]
                            current_text = new_text
                            text_area.delete(1.0, tk.END)
                            text_area.insert(tk.END, new_text)
                            

                        else:
                            text_area.insert(tk.END, f"{predicted_character}")
                        prediction_time = None
                else:
                    last_predicted_char = predicted_character
                    prediction_time = None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(img)

            video_label.img_tk = img_tk
            video_label.config(image=img_tk)

            update_job = video_label.after(34, update_frame)

        update_frame()

    def back_to_main_menu():
        stop_SL()
        new_window.destroy()
        create_main_menu()

    new_window.bind("<Escape>", lambda e: back_to_main_menu())
    new_window.protocol("WM_DELETE_WINDOW", lambda: (stop_SL(), new_window.destroy()))

    label = tk.Label(new_window, text="Sign Language to text", font=("Arial", 30))
    label.place(x=550, y=50, anchor="w")  
    label.config(padx=5, pady=5) 

    button_start = CTkButton(
        master=new_window, 
        text="Start",
        corner_radius = 32, 
        fg_color="#00D100", 
        hover_color="#00A7E1", 
        text_color="#FFFFFF",
        font=("Arial", 20), 
        border_width=2, 
        command=start_SL
    )
    button_start.place(x=135, y=160)

    button_back = CTkButton(
        master=new_window, 
        text="Back",
        corner_radius = 32, 
        fg_color="#003459", 
        hover_color="#00A7E1", 
        text_color="#FFFFFF",
        font=("Arial", 20), 
        border_width=2, 
        command=back_to_main_menu
    )
    button_back.place(x=50, y=400) 

    text_label = tk.Label(new_window, text="Output:", font=("Arial", 18))
    text_label.place(x=65, y=300)

    text_area = tk.Text(new_window, wrap=tk.WORD, font=("Arial", 16), width=30, height=5)
    text_area.place(x=65, y=350) 

    new_window.mainloop()

create_main_menu()
