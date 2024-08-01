from diffusion.src.logic import get_txt_to_img_pipeline, generate_image, clear_cache, get_parser
from dotenv import load_dotenv
from telebot.types import InputFile
from queue import Queue

import telebot
import threading
import time
import os
import io
import shlex

#  parser.add_argument('command', choices=["/generate"])
#     parser.add_argument('-p', '--prompt', type=str, required=True)
#     parser.add_argument('-n', '--negprompt', type=str, required=False, default="")
#     parser.add_argument(
#         '-o', '--orientation', type=str, choices=['landscape', 'portrait', 'square'], required=False, default="square"
#         )
#     parser.add_argument('-s', '--steps', type=int, choices=[10 * i for i in range(3, 10)], required=False, default=60)

load_dotenv()
pipeline = get_txt_to_img_pipeline()
parser = get_parser()
bot = telebot.TeleBot(os.getenv("BOT_API"))
bot.pipeline = pipeline
queue = Queue()
event = threading.Event()

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
            execute_task(bot, msg, args)
            time.sleep(0.1)

answerer=threading.Thread(target=answer_from_queue)
answerer.start()


@bot.message_handler(commands = ["generate"])
def generate_telebot(msg):
    try:
        args = parser.parse_args(shlex.split(msg.text))
        currsize = queue.qsize()
        queue.put(args, msg)
        bot.reply_to(msg, f"Job accepted. Please wait, there are currently {currsize} items in queue")
    except Exception as e:
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