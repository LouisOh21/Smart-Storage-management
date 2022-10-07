# reading the input using the camera
import cv2
import time
import os
from PIL import Image
import joblib
from skimage.io import imread
from skimage.transform import resize
import keras

iter = 1
cam_port = 0

def resize_all(src, pklname, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary,
    together with labels and metadata. The dictionary is written to a pickle file
    named '{pklname}_{width}x{height}px.pkl'.

    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """

    height = height if height is not None else width

    data = dict()
    data['description'] = 'resized ({0}x{1}) in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    pklname = f"{pklname}_{width}x{height}px.pkl"
    print(os.listdir(src))
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        current_path = os.path.join(src, subdir)

        for file in os.listdir(current_path):
            if file[-3:] in {'jpg', 'png'}:
                im = imread(os.path.join(current_path, file))
                im = resize(im, (width, height))  # [:,:,::-1]
                data['label'].append(subdir)
                data['filename'].append(file)
                data['data'].append(im)

_is_tensorflow = True
while True:
    print("Testing AI")
    try:
        if _is_tensorflow == False:
            model = joblib.load('cnn_model.pkl')
        else:
            model = keras.models.load_model("cnn_image")
        cam = cv2.VideoCapture(cam_port)
        result, image = cam.read()
        #cv2.imshow('frame', image)
        result = cv2.imwrite(r"C:\temp\img_test/"+str(iter)+".png" , image)
        if result == True:
            print("File saved successfully")
        else:
            print("Error in saving file")

        width = 80
        height = 80
        image = resize(image, (width, height))
        print(image.shape)
        print(model.predict(image.reshape(1,80,80,3)))
    except Exception as e:
        print("Error : ", e)

    time.sleep(5)
    iter += 1

