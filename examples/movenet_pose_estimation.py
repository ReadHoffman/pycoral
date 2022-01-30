# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet

Example usage:
```
bash examples/install_requirements.sh movenet_pose_estimation.py

python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""
import os
import os.path as path
# import argparse
import time
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import cv2
import numpy as np
from math import atan2, degrees

key_point_ref = {
0:'nose',
1:'leftEye',
2:'rightEye',
3:'leftEar',
4:'rightEar',
5:'leftShoulder',
6:'rightShoulder',
7:'leftElbow',
8:'rightElbow',
9:'leftWrist',
10:'rightWrist',
11:'leftHip',
12:'rightHip',
13:'leftKnee',
14:'rightKnee',
15:'leftAnkle',
16:'rightAnkle',
17:'centerHip',
18:'centerShoulder',
19:'centerKnee',
20:'centerAnkle'
}

point_line_ref = [
    [18,0],
    [18,7],
    [18,8],
    [18,17],
    [17,19],
    [19,20]
    ]


keypoints = [0,7,8,17,18,19,20]

def get_mid_pct(leftHipList,rightHipList):
    data = np.array([leftHipList, rightHipList])
    return np.average(data, axis=0)

def update_keypoints_xy(pose,img_width,img_height):
    new_pose_list = np.zeros(shape=(len(pose),5),dtype=float)
    for i, row in enumerate(pose):
        x = int(row[1] * img_width)
        y = int(row[0] * img_height)
        new_pose_list[i] = np.append([[row]],[[x,y]])
    return new_pose_list

def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def angle_btw_two_pts(coord1,vertex,coord2):
    a = np.array([coord1[0],coord1[1]])
    b = np.array([vertex[0],vertex[1]])
    c = np.array([coord2[0],coord2[1]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


_NUM_KEYPOINTS = 17

ROOT_DIR = path.abspath(path.join(__file__ ,"../.."))
# MODEL_FILE = 'test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite'
MODEL_FILE = 'test_data/movenet_single_pose_thunder_ptq_edgetpu.tflite'  #slower more accurate
MODEL_PATH = os.path.join(ROOT_DIR, MODEL_FILE)


cap = cv2.VideoCapture(0)


def main():
    print("running program")
    print("using the following model: ",MODEL_PATH)

    pTime = time.time()
    interpreter = make_interpreter(MODEL_PATH) # fix this
    interpreter.allocate_tensors()
    
    nloops = 0
    while nloops<150:
        #cv2 image
        success, img = cap.read()
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.rotate(img,cv2.ROTATE_180)

        resized_img = cv2.resize(img, common.input_size(interpreter), Image.ANTIALIAS)
        common.set_input(interpreter, resized_img)

        interpreter.invoke()

        pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
        hip_center = get_mid_pct(pose[11],pose[12])
        shoulder_center = get_mid_pct(pose[5],pose[6])
        knee_center = get_mid_pct(pose[13],pose[14])
        ankle_center = get_mid_pct(pose[15],pose[16])
        pose = np.append(pose,[hip_center,shoulder_center,knee_center,ankle_center], axis=0)
#         pose = pose.round(decimals=4, out=None)
        image_from_array = Image.fromarray(img)
        width, height = image_from_array.size
        pose = update_keypoints_xy(pose,width,height)
        print(pose)

        for point_id in keypoints:
            x1 = int(pose[point_id][1] * width)
            y1 = int(pose[point_id][0] * height)
            cv2.circle(img,(x1,y1),3,(255,0,0),5)
            cv2.putText(img, key_point_ref[point_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        
        for i1, i2 in point_line_ref:
            x1 = int(pose[i1][1] * width)
            y1 = int(pose[i1][0] * height)
            x2 = int(pose[i2][1] * width)
            y2 = int(pose[i2][0] * height)
            cv2.line(img,(x1,y1),(x2,y2),(0, 0, 0),3)
        
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        nloops += 1
        #img.save(args.output)
        #print('Done. Results saved at', args.output)
        
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

