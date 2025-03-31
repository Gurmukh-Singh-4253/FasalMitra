#!/usr/bin/env python

import logging

import tensorflow as tf
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import google.generativeai as genai

APIKEY = "AIzaSyDNfZzv2PrddjWnHDhWOhVCtWSAN8pV5EU"
genai.configure(api_key=APIKEY)

model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}, Welcome to the Fasal Mitra Bot!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("I am the Telegram bot for Fasal Mitra. Send an image to get analysis on the crop.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    response = chat.send_message(update.message.text)
    await update.message.reply_text(response.text)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Recieved the image, predicting disease...")
    # Pass image to model

async def recommend_fertilizer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    NModel = tf.keras.model.load("Nitrogen_model.h5")
    PModel = tf.keras.model.load("Phosphorus_model.h5")
    KModel = tf.keras.model.load("Potassium_model.h5")
    print(NModel.input_shape)
    print(PModel.input_shape)
    print(KModel.input_shape)
    NModel.predict()
    PModel.predict()
    KModel.predict()
    pass

def main() -> None:
    """Start the bot."""
    application = Application.builder().token("8022688398:AAGmmuEbTh13o0-IP_1fnDdRRpU8o_vcAbM").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
