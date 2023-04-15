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
        # print(f"Mask Photo Shape: {photo_mask.shape} Original Photo Shape: {y.shape}")

        # print(photo_mask.shape, background_mask.shape, image.shape, name)
        # cv2.imwrite(f"remove_bg/{name}.png", photo_mask)
        # # print(type(photo_mask))

        # # cv2.imwrite(f"remove_bg/{name}.png", photo_mask*255)
        # # cv2.imwrite(f"remove_bg/{name}.png", background_mask*255)

        # # cv2.imwrite(f"remove_bg/{name}.png", image * photo_mask)
        # # cv2.imwrite(f"remove_bg/{name}.png", image * background_mask)
        # masked_photo = image * photo_mask
        # background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
        # background_mask = background_mask * [0, 0, 0]
        # final_photo = masked_photo + background_mask
        # # Convert the masked photo to RGBA mode (with transparency)
        # masked_photo = cv2.cvtColor(final_photo, cv2.COLOR_BGR2RGBA)
        # # Get the pixel data of the masked photo
        # pixels = masked_photo.reshape((-1, 4))
        # # Create a new list of pixels with a transparent background
        # new_pixels = []
        # for pixel in pixels:
        #     # If the pixel is not already transparent
        #     if pixel[0] != 0 or pixel[1] != 0 or pixel[2] != 0:
        #         # Set the pixel to have 0 (fully transparent) alpha
        #         new_pixels.append((pixel[0], pixel[1], pixel[2], 0))
        #     else:
        #         # Keep the transparent pixel as-is
        #         new_pixels.append(pixel)
        # # Update the masked photo with the new pixel data
        # masked_photo = np.array(new_pixels, dtype=np.uint8).reshape(final_photo.shape)
        # # Save the masked photo with transparent background
        # cv2.imwrite(f"remove_bg/{name}.png", masked_photo)


        masked_photo = image * photo_mask
        # background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
        # background_mask = background_mask * [255, 255, 0]
        # background_mask = cv2.cvtColor(background_mask, cv2.COLOR_BGR2RGBA)
        # final_photo = masked_photo + background_mask

        # print(f"mask Photo {masked_photo.shape} - background masked {background_mask.shape}")


        # ### Test-I
        # background_mask = cv2.cvtColor(background_mask, cv2.COLOR_BGR2RGBA)
        # # Get the pixel data of the masked photo
        # pixels = background_mask.reshape((-1, 4))
        # # Create a new list of pixels with a transparent background
        # new_pixels = []
        # for pixel in pixels:
        #     # If the pixel is not already transparent
        #     if pixel[0] != 0 or pixel[1] != 0 or pixel[2] != 0:
        #         # Set the pixel to have 0 (fully transparent) alpha
        #         new_pixels.append((pixel[0], pixel[1], pixel[2], 0))
        #     else:
        #         # Keep the transparent pixel as-is
        #         new_pixels.append(pixel)
        # # Update the masked photo with the new pixel data
        # background_mask = np.array(new_pixels, dtype=np.uint8).reshape(background_mask.shape)

        cv2.imwrite(f"{args.dir_path}/{name}.png", masked_photo)
        print(f"Photo {name}.png saved!!")
