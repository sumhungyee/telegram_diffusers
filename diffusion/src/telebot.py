from src.logic import get_txt_to_img_pipeline, generate_image, clear_cache, get_parser, replace_curly_quotes
from dotenv import load_dotenv
from telebot.types import InputFile
from queue import Queue

import telebot
import threading
import time
import os
import io
import shlex



load_dotenv()
pipeline = get_txt_to_img_pipeline()

bot = telebot.TeleBot(os.getenv("BOT_API"))
bot.pipeline = pipeline
queue = Queue()
event = threading.Event()
bot.in_session = False

def execute_task(bot, msg, args):
    prompt = args.prompt
    negprompt = args.negprompt
    orientation = args.orientation
    steps = args.steps
    image = generate_image(pipeline, prompt, negative_prompt=negprompt, image_type=orientation, num_inference_steps=steps)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    file = io.BytesIO(img_byte_arr)
    bot.send_photo(msg.chat.id, InputFile(file), has_spoiler=True)


def answer_from_queue():
    while not event.is_set():
        if queue.qsize() >= 1:
            args, msg = queue.get()
            bot.in_session = True
            execute_task(bot, msg, args)
            clear_cache()
            bot.in_session = False
            time.sleep(0.1)

answerer=threading.Thread(target=answer_from_queue)
answerer.start()


@bot.message_handler(commands = ["generate"])
def generate_telebot(msg):
    parser = get_parser()
    msg.text = replace_curly_quotes(msg.text)
    try:
        args = parser.parse_args(shlex.split(msg.text))
        currsize = queue.qsize()
        queue.put((args, msg))

        if bot.in_session:
            to_append = " and currently processing one image."
        else:
            to_append = "."

        bot.reply_to(msg, f"Job accepted. Please wait, there are currently {currsize} items in queue" + to_append)
    except:
        bot.reply_to(msg, "Request failed, format your prompt properly. See /help for a list of arguments")

@bot.message_handler(commands = ["help"])
def get_help(msg):
    help_str = """To generate an image, use /generate with CLI-style arguments.
    Available arguments:
    --prompt, or -p (mandatory). Expected type is a string. Use quotation marks (' or ").
    --negprompt, or -n (optional), defaults to empty.
    --orientation, or -o (optional), defaults to square image. Available: landscape, portrait, square (lowercase)
    --steps, or -s (optional), defaults to 60.
    Example usage: /generate -p 'A big chocolate bar' --negprompt "monochrome" -s 50
    """
    bot.reply_to(msg, help_str)


bot.infinity_polling(timeout = 10, long_polling_timeout=5)