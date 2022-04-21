import random
import numpy as np
from grabscreen import grab_screen
import cv2
import time
from constants import IMAGE_WIDTH,IMAGE_HEIGHT,NUM_KEYS,W_VEC,A_VEC,S_VEC,D_VEC,WA_VEC,WD_VEC,SA_VEC,SD_VEC,NK_VEC,W_HEX,A_HEX,S_HEX,D_HEX
from keys import PressKey, ReleaseKey
from getkeys import key_check
from testing_utils import *
from tensorflow import keras
import tensorflow as tf
import sys

W = 0
S = 1
A = 2
D = 3
WA = 4
WD = 5
SA = 6
SD = 7
NK = 8

def main():

    paused = False
    # load model
    model_path = sys.argv[1]
    network = tf.keras.models.load_model(model_path)
    print("Starting in")
    for i in list(range(5))[::-1] :
        print(i)
        time.sleep(1)
    while True:
        start = time.time()
        if not paused:
            # Get screenshot
            img = grab_screen((0,0,800,600))
            img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
            img = cv2.resize(img,(480,360))
            input_img = np.expand_dims(np.array(list(img / 255.0),dtype=np.float),0)
            # Get network prediction
            output_key = list(np.zeros((NUM_KEYS,),dtype=np.int))
            prediction = network.predict(input_img)
            #np.array([4.5, 0.1, 0.1, 0.1, 1.8, 1.8, 0.5, 0.5, 0.2])
            output_key = np.argmax(prediction)
            #output_key[prediction] = 1

            # Send output
            if output_key == W:
                straight()
            elif output_key == A:
                if random.randrange(0,3) == 1 :
                    acc_left()
                else :
                    left()
            elif output_key == S:
                brake()
            elif output_key == D:
                if random.randrange(0,3)  == 1 :
                    acc_right()
                else:
                    right()

            elif output_key == WA:
                acc_left()
            elif output_key == WD:
                acc_right()
            elif output_key == SA:
                reverse_left()
            elif output_key == SD:
                reverse_right()
            elif output_key == NK :
                if random.randrange(0,4) == 1:
                    straight()
                else:
                    do_nothing()


        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)

            else:
                print('Pausing!')
                paused = True
                ReleaseKey(A_HEX)
                ReleaseKey(W_HEX)
                ReleaseKey(D_HEX)
                time.sleep(1)



if __name__=='__main__':
    main()