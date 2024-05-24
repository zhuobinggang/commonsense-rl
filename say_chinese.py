from gtts import gTTS
import os
import playsound

def speak(text):
    tts = gTTS(text=text, lang='zh-TW')
    filename = "abc.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

