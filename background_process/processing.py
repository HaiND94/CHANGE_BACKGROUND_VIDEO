import json
import random
import os
import cv2
import logging
import numpy as np
import io
import traceback
import argparse

from PIL import Image

try:
    from object_segment import load_model, get_object
except:
    from .object_segment import load_model, get_object


IMG_EXTEND = ['jpg', 'png', 'jpeg']
VID_EXTEND = ['mp4', 'avi', 'mov']


logging.basicConfig(level=logging.INFO)


def background_img(net, image, transprent=False, green=False, black=False):
    """Decreption: To make new image follow require
    :param - net: model u2net
    :param - image: path of image
    :param - green: mode image
    :param - transprent: mode image
    :param - black: mode image
    result: image follow require
    """

    try:
        pred, pred_blur, black_img, green_img, transprent_img = get_object(img=image, net=net, debug=False)
    except Exception as e:
        logging.error(f"Can not render image \n {e}")
        return False

    if transprent:
        return transprent_img
    elif green:
        return green_img
    elif black:
        return black_img
    else:
        return False
        


def background_vid(net, video, output, transprent=False, green=False, black=False, debug=False):
    """Decreption: To make new video follow require
    :param - net: model u2net
    :param - image: path of video
    :param - green: mode video
    :param - transprent: mode video
    :param - black: mode video
    result: video follow require
    """

    if not os.path.isfile(video):
        logging.error("Can not find video")
    
    # Init video code 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        logging.error("Error opening video stream or file")
        return False
    
    # Get info video input
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  cap.get(cv2.CAP_PROP_FPS)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    length = int(cv2.VideoCapture.get(cap, property_id))

    logging.info(f"Fame of video: {fps}\n Size: {width}*{height}\n Total of frame: {length}")

    # Output video
    out = cv2.VideoWriter(output, fourcc, fps, (int(width),int(height)))

    # Read until video is completed
    i = 0
    while(cap.isOpened()):
        i += 1

        print("Processing frame %4d/%4d"%(i, length), end="\r")
    # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            #  Convert image to buffer
            is_success, buffer = cv2.imencode(".PNG", frame)
            if is_success:
                image_buff = io.BytesIO(buffer)
            else:
                logging.warn("Can not save image frame to buffer")
                return False
            

            im_new = background_img(net, image_buff, transprent, green, black)
            if not im_new: 
                logging.warn("Can not remove background image")
                return False
            # im_new.save("./dev_env/test_result.PNG", format="PNG")
            # Convert image pillow to cv2 format
            cv2_img = cv2.cvtColor(np.array(im_new), cv2.COLOR_RGB2BGR)
            
            # break
            # Display the resulting frame
            if debug:
                cv2.imshow('Frame',cv2_img)

            out.write(cv2_img)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    out.release() 
    
    # Closes all the frames
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    video = "/home/haind/Documents/VFAST/Background_removal/dev_env/mc_nanao_crop.mp4"
    model_path = "/home/haind/Documents/VFAST/Background_removal/model/u2net_human_seg.pth"
    net = load_model(model_path, 'u2net')

    background_vid(net, video, green=True)
    