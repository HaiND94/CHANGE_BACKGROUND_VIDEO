import argparse
import logging
import traceback

from background_process import background_img, background_vid, load_model
from PIL import Image

IMG_EXTEND = ['jpg', 'png', 'jpeg']
VID_EXTEND = ['mp4', 'avi', 'mov']


parser = argparse.ArgumentParser(description='This program to remove background for image or video.')

parser.add_argument('-model', required=True, type=str,
                    help='input model')
parser.add_argument('-i', required=True, type=str,
                    help='input file')
parser.add_argument('-o', type=str, default="output",
                    help='Set to mode render output file, just can be render to avi file or png image')
parser.add_argument('-transprent', type=bool, default=False,
                    help='Set to mode render output file, just can be render to avi file or png image')
parser.add_argument('-green', type=bool, default=False,
                    help='Set to mode render output file, just can be render to avi file or png image')
parser.add_argument('-dark', type=bool, default=False,
                    help='Set to mode render output file, just can be render to avi file or png image')
    
args = parser.parse_args()

# Get input info
input = args.i
exe = input.split('.')[-1].lower()

# Detect mode run in image or video
video_mode = True

if exe in IMG_EXTEND:
    video_mode = False
    output_file = args.o + ".png"
elif exe in VID_EXTEND:
    output_file = args.o + ".avi"
else:
    raise(f"Input file must be in {VID_EXTEND} or {IMG_EXTEND}")

# Get mode render
transprent = args.transprent
green = args.green
black = args.dark

# Get path model
model_path = args.model
try:
    net =load_model(model_path)
except Exception:
    traceback.print_exc()

# Start process
if not video_mode:
    try:
        im = background_img(net=net, image=input, transprent=transprent, green=green, black=black)
        im.save(output_file, format='PNG')
    except Exception:
        traceback.print_exc()
else:
    try:
        background_vid(net, input, output_file, transprent=transprent, green=green, black=black)
    except Exception:
        traceback.print_exc()
        
logging.info("Finished!")

