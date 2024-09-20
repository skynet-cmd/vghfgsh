import openai
import pyaudio
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import pygame
import noisereduce as nr
import numpy as np
import uuid  # For generating unique file names
import re  # For flexible language request detection

# Set up OpenAI API key
openai.api_key = ''

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize Google Translate API
translator = Translator()

# Initialize pygame for playing audio
pygame.mixer.init()

# Global variables to store conversation history, language preference, and practice mode state
conversation_history = []
preferred_language = None
practice_mode = False
learning_language = None
current_step = 0

# Language code mapping for gTTS, including Albanian
LANGUAGE_CODE_MAP = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "albanian": "sq",  # Added Albanian (Shqip)
    # Add more languages if needed
}

# Sample vocabulary and phrases for each language
LEARNING_SESSIONS = {
    "spanish": {
        "vocabulary": ["hola", "adiós", "gracias", "por favor", "gato", "perro"],
        "phrases": ["Buenos días", "¿Cómo estás?", "Me llamo..."]
    },
    "albanian": {
        "vocabulary": ["përshëndetje", "mirupafshim", "faleminderit", "të lutem", "mace", "qen"],
        "phrases": ["Mirëmëngjes", "Si je?", "Emri im është..."]
    },
    # Add more languages and phrases
}

# Function to reduce background noise from the audio data
def reduce_noise(audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    reduced_noise = nr.reduce_noise(y=audio_np, sr=16000)
    return reduced_noise.tobytes()

# Function to recognize speech with more time for input
def recognize_speech(language_code="en-US"):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for input...")
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)  # Increased timeout
        audio_data = audio.get_raw_data()
        filtered_audio = reduce_noise(audio_data)
        filtered_audio_obj = sr.AudioData(filtered_audio, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
        
        try:
            text = recognizer.recognize_google(filtered_audio_obj, language=language_code)
            print(f"Recognized text: {text}")
            return text
        except sr.UnknownValueError:
            print("I couldn't clearly recognize what you said.")
            return ""
        except sr.RequestError:
            print("Error with the Speech Recognition service.")
            return ""

# Function to generate a response using GPT-3.5 Turbo
def generate_gpt_response(prompt, language):
    try:
        conversation_history.append({"role": "user", "content": prompt})
        messages = [{"role": "system", "content": f"You are a helpful assistant. Respond in {language}."}] + conversation_history
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        gpt_response = response['choices'][0]['message']['content']
        conversation_history.append({"role": "assistant", "content": gpt_response})
        return gpt_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

# Function to check if the user wants to start a learning session
def check_learning_request(user_input):
    if "i want to start learning" in user_input.lower():
        match = re.search(r"i want to start learning (\w+)", user_input.lower())
        if match:
            return match.group(1)  # Return the language they want to learn
    return None

# Function to convert text to speech using gTTS (Google Text-to-Speech)
def gtts_speak_text(text, language="en"):
    language_code = LANGUAGE_CODE_MAP.get(language.lower(), "en")
    
    # Generate a unique file name to avoid overwriting
    unique_filename = f"response_{uuid.uuid4()}.mp3"
    save_path = os.path.join(os.path.expanduser("~"), unique_filename)
    
    try:
        # Generate the speech in the specified language
        tts = gTTS(text=text, lang=language_code)
        tts.save(save_path)
        
        # Play the audio using pygame
        pygame.mixer.music.load(save_path)
        pygame.mixer.music.play()
        
        # Wait for the speech to finish
        while pygame.mixer.music.get_busy():
            continue
        
        # Remove the file after playing
        pygame.mixer.music.stop()  # Ensure playback is fully stopped
        os.remove(save_path)  # Safely remove the file
    except ValueError as e:
        print(f"Error generating speech: {e}. Language '{language}' is not supported.")
    except PermissionError as e:
        print(f"Permission error: {e}. Try running the script with appropriate file access permissions.")

# Practice mode: Guide the user through vocabulary and exercises
def practice_mode_session(language):
    global current_step
    if language not in LEARNING_SESSIONS:
        gtts_speak_text(f"Sorry, I don't have lessons for {language}.", 'en')
        return

    # Fetch vocabulary and phrases for the selected language
    vocab_list = LEARNING_SESSIONS[language]["vocabulary"]
    phrase_list = LEARNING_SESSIONS[language]["phrases"]

    if current_step < len(vocab_list):
        # Introduce vocabulary words and ask for repetition
        word = vocab_list[current_step]
        gtts_speak_text(f"Repeat after me: {word}", language)
    elif current_step < len(vocab_list) + len(phrase_list):
        # Move to phrases after vocabulary
        phrase = phrase_list[current_step - len(vocab_list)]
        gtts_speak_text(f"Now try this phrase: {phrase}", language)
    else:
        gtts_speak_text("Good job! You've completed this lesson.", 'en')
        current_step = 0  # Reset the lesson

    current_step += 1

# Main function to handle conversation flow
def conversation():
    global preferred_language, practice_mode, learning_language, current_step
    print("Starting conversation. Say 'exit' to quit or specify a language (e.g., 'Let's speak in English').")
    
    while True:
        if preferred_language:
            lang_code = {'en': 'en-US', 'es': 'es-ES', 'fr': 'fr-FR'}.get(preferred_language, 'en-US')
        else:
            lang_code = 'en-US'

        user_input = recognize_speech(language_code=lang_code)
        
        if user_input.lower() == "exit":
            print("Ending conversation.")
            break

        # Check if the user wants to start a learning session
        learning_request = check_learning_request(user_input)
        if learning_request:
            learning_language = learning_request
            practice_mode = True
            gtts_speak_text(f"You want to learn {learning_language}. Let's get started!", 'en')
            current_step = 0  # Reset the step counter for the lesson
            continue

        # If in practice mode, go through the learning session
        if practice_mode:
            practice_mode_session(learning_language)
            continue

        # Normal conversation mode
        if not preferred_language:
            preferred_language = "english"  # Default to English if no language is set
        gpt_response = generate_gpt_response(user_input, preferred_language)
        gtts_speak_text(gpt_response, preferred_language)

# Run the conversation
if __name__ == "__main__":
    conversation()
