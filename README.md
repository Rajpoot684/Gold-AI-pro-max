import streamlit as st
from utils import astro_signal
from voice import speak

st.set_page_config(page_title="Gold AI Pro Max", layout="centered")

st.title("💰 Gold AI Pro Max – MacBook M2 Version")
st.markdown("🚀 AI-powered Gold (XAU/USD) prediction with Astrology & Voice")

# Show signal
astro = astro_signal()
st.success(f"🪐 Astrology Signal: {astro}")

# Speak it
if st.button("🔊 Speak Signal"):
    speak(f"Astrology signal is {astro}")
    def astro_signal():
    # Dummy signal – you can replace with your own logic later
    return "Bullish"
    import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    from telegram.ext import Updater, CommandHandler

TOKEN = "YOUR_BOT_TOKEN"

def signal(update, context):
    update.message.reply_text("🔔 Astrology Signal: Bullish")

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("signal", signal))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
    streamlit
yfinance
pandas
numpy
scikit-learn
matplotlib
tensorflow-macos
tensorflow-metal
ta
pyttsx3
python-telegram-bot==13.7
# Gold AI Pro Max – MacBook M2
Streamlit-based AI Gold Predictor with Astrology & Voice
- Run on macOS M1/M2 with tensorflow-metal
- Voice Alerts
- Telegram Bot ready

## Run
