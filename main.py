#!/usr/bin/env python3

import os
import sys
import logging

import classifier as ic
from telegram import Update
from telegram.ext import Application, CommandHandler, ConversationHandler, ContextTypes, MessageHandler, filters


async def start_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('Por favor, envía una imagen que te gustaría clasificar')
    return 0


async def process_user_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    imgfile = await update.message.photo[-1].get_file()
    await imgfile.download('image.jpg')
    await update.message.reply_text("¡He recibo tu imagen! Espera mientras la proceso...")

    classifier = ic.ImageClassifier('./image.jpg')
    classifier.load_resized_image()
    classifier.load_classifier_model('./model')

    prediction = classifier.predict()
    prediction = classifier.to_spanish(prediction)

    await update.message.reply_text(
        f'Parece que tu imagen es un {prediction}. '
        '¡Espero haber acertado!'
    )

    return ConversationHandler.END


async def help_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Escribe el comando /start')


async def cancel_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('¡Hasta luego! Espero que nos veamos pronto...')
    return ConversationHandler.END

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if token is None:
        logging.error(f'TELEGRAM_BOT_TOKEN environment variable is required')
        sys.exit(1)

    app = Application.builder().token(token).build()

    conversation = ConversationHandler(
        entry_points=[CommandHandler('start', start_command_handler)],
        states={
            0: [MessageHandler(filters.PHOTO, process_user_image), CommandHandler('skip', cancel_command_handler)]
        },
        fallbacks=[CommandHandler('cancel', cancel_command_handler)]
    )

    app.add_handler(conversation)
    app.run_polling()
