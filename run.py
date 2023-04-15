import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2 
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description ="Pass the arguments")
    parser.add_argument("--dir_path", default='outbg')
    parser.add_argument("--model_path", default="")
    parser.add_argument("--data_path", default="",help="Image folder path")
    args = parser.parse_args()
    return args

""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":

    args = parse_args()
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir(args.dir_path)
    print("Sucessfully Created Dir...")

    """ Loading model: DeepLabV3+ """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("model.h5")

    print("Model Loaded....") 
    for i in os.listdir(args.data_path):
        name = (i.split(".")[0])
        path = args.data_path + i

        """ Read the image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)
        y = y > 0.5

        photo_mask = y
        background_mask = np.abs(1-y)


        masked_photo = image * photo_mask

        cv2.imwrite(f"{args.dir_path}/{name}.png", masked_photo)
        print(f"Photo {name}.png saved!!")
