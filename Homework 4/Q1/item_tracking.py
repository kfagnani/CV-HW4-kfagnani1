import cv2
import numpy as np
import numpy
from matplotlib import pyplot as plt
import tensorflow
import camera_matrix_calculation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_data = ImageDataGenerator()

gen = img_data.flow_from_directory("./item_imgs",target_size=(200,200))

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, Dropout




inside = Input(shape=(200,200,3))
c_one = Conv2D(20, 3, activation="relu")(inside)
mp_one = MaxPooling2D(2)(c_one)
c_two = Conv2D(40, 3, activation="relu")(mp_one)
mp_two = MaxPooling2D(2)(c_two)
c_three = Conv2D(60, 3, activation="relu")(mp_two)
mp_three = MaxPooling2D(2)(c_three)
c_four = Conv2D(80, 3, activation="relu")(mp_three)
mp_four = MaxPooling2D(2)(c_four)
flat = Flatten()(mp_four)
d_one = Dense(800, activation="relu")(flat)
d_two = Dense(600, activation="relu")(d_one)
d_three = Dense(350, activation="relu")(d_two)
d_four = Dense(100, activation="relu")(d_three) 
drop_one = Dropout(.2)(d_four)
d_five = Dense(50, activation="relu")(drop_one)
outside = Dense(2, activation="softmax")(d_five)

model = Model(inside, outside)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(gen, epochs=10)

filename ="kirstyn.mov"
capture = cv2.VideoCapture(filename)


def getClassName(index):
    if index == 0:
        return "object found"
    else:
        return "object not found"

font = cv2.FONT_HERSHEY_COMPLEX
org = (50, 50)
cam_matx = camera_matrix_calculation.camera_matx

cam_matx_inv = np.linalg.inv(cam_matx)

img_found = cv2.imread("./item_imgs/found/capture_isp_1.png")
res,coords = model.predict(np.array([img_found]))

(fX, fY, fW, fH) = coords

size = "(" + str(fW) + "," + str(fH) + ")"

dimension_matrix = cam_matx_inv.dot(project_points)
size = "(" + str(-1*dimension_matrix[0][0]) + "," + str(-1*dimension_matrix[1][0]) + ")"

fontScale = 1
color = (255, 0, 0)  
thickness = 2

def img_alignment(img1, img2):
    img1, img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
    img_size = img1.shape
    warp_mode = cv2.MOTION_TRANSLATION

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3,dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3,dtype=np.float32)
    
    n_iterations = 5000
    termination_eps = 1e-10

    criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, n_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(img2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img2_aligned = cv2.warpAffine(img2, warp_matrix, (img_size[1], img_size[0]), flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return img2_aligned

while True:
    _, img1 = capture.read()
    _, img2 = capture.read()

    res = model.predict(np.array([cv2.resize(img1, (200,200))]))
    img1 = cv2.putText(img1, getClassName(res.argmax(axis=1)) , org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    if getClassName(res.argmax(axis=1))=="object found":
        img1 = cv2.putText(img1,size,(100,100), font,fontScale, color, thickness, cv2.LINE_AA)

    diff = cv2.absdiff(img1, img2)
    
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff_blur = cv2.GaussianBlur(diff_gray, (5,5,), 0)

    _, binary_img = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, b, l = cv2. boundingRect(contour)
        if cv2.contourArea(contour) > 300:
            cv2.rectangle(img1, (x, y), (x+b, y+l), (0,255,0), 2)
    
    cv2.imshow("Motion", img1)
    key = cv2.waitKey(1)
    if key%256 == 27:
        print("Closing program")
        exit()