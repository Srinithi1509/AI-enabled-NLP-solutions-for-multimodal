import streamlit as st
import sqlite3
import speech_recognition as sr
import pyttsx3
import pytesseract
import smtplib
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from googletrans import Translator
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from playsound import playsound
from pdfminer.high_level import extract_text
import docx

# ==========================
# 🔹 DATABASE FUNCTIONS
# ==========================
def create_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()
    st.success("User created successfully.")

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None

# ==========================
# 🔹 TRANSLATION FUNCTION
# ==========================
def translate_text(text, target_language):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.write(f"Error in translation: {e}")
        return text

# ==========================
# 🔹 AUDIO RECOGNITION
# ==========================
def audio_to_text_from_microphone(target_language):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return translate_text(text, target_language)
        except:
            st.write("Could not understand audio.")
            return None

# ==========================
# 🔹 IMAGE TEXT EXTRACTION
# ==========================
def image_to_text(image, target_language):
    text = pytesseract.image_to_string(image)
    st.write(f"Extracted Text: {text}")
    return translate_text(text, target_language)

# ==========================
# 🔹 DOCUMENT TEXT EXTRACTION
# ==========================
def extract_text_from_document(file, target_language):
    text = ""
    if file.type == "text/plain":
        text = file.read().decode("utf-8")
    elif file.type == "application/pdf":
        text = extract_text(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    
    st.write("Extracted Text:")
    st.text_area("Original Text", text, height=200)
    
    translated_text = translate_text(text, target_language)
    st.text_area("Translated Text", translated_text, height=200)

    # Download button for translated text
    st.download_button(label="Download Translated Text", data=translated_text, file_name="translated_text.txt", mime="text/plain")

    return translated_text

# ==========================
# 🔹 MAIN FUNCTION
# ==========================
def main():
    st.set_page_config(page_title="AI Multi-Input Recognition", layout="wide")

    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False

    # Sidebar for Authentication
    st.sidebar.title("Authentication")
    menu = ["Home", "Sign Up", "Login"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("Welcome to AI Multi-Input Recognition System")
        st.write("Recognize hand signs, audio, images, and documents.")

    elif choice == "Sign Up":
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        if st.button("Sign Up"):
            create_user(new_username, new_password)

    elif choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.user_authenticated = True
                st.success("Login Successful")
            else:
                st.error("Invalid Credentials")

    if st.session_state.user_authenticated:
        st.subheader("Select an Input Mode:")
        input_choice = st.selectbox("Choose an option", ["Hand Sign", "Audio", "Image", "Document"])

        target_language = st.selectbox("Translate to:", ["en", "es", "fr", "de", "hi", "ta", "ml", "te"])

        if input_choice == "Hand Sign":
            st.write("Hand Sign Recognition is under development.")

        elif input_choice == "Audio":
            st.write("Listening...")
            translated_text = audio_to_text_from_microphone(target_language)
            st.write(f"Translated Text: {translated_text}")

        elif input_choice == "Image":
            image_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
            if image_file:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded Image")
                translated_text = image_to_text(img, target_language)
                st.write(f"Translated Text: {translated_text}")

        elif input_choice == "Document":
            document_file = st.file_uploader("Upload a Document (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])
            if document_file:
                translated_text = extract_text_from_document(document_file, target_language)

if __name__ == "__main__":
    main()
