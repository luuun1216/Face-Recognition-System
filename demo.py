import camera_capture
import cv2
import os
import time
import torch
import numpy as np
from inception_resnet_v1 import InceptionResnetV1
# load model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# helper function
def encode(img):
    # print(type(img))
    res = resnet(torch.Tensor(img))
    return res

# get encoded features for all saved images
saved_pictures = "../save"
all_people_faces = {}
for file in os.listdir(saved_pictures):
    if file.endswith(".jpg"):
        person_face, _ = os.path.splitext(file)
        img = cv2.imread(os.path.join(saved_pictures, file))
        # cropped = mtcnn(img)
        cropped = img
        reshaped_array = np.transpose(cropped, (2, 0, 1))
        reshaped_array = np.expand_dims(reshaped_array, axis=0)
        if reshaped_array is not None:
            all_people_faces[person_face] = encode(reshaped_array)[0, :]

if camera_capture.open_input_source("0"):
# if camera_capture.open_input_source("../test.mp4"):
    thres=0.7
    while True:
    # start_time = time.time()
    # for i in range(1):
        
        result = camera_capture.process_image()
        original_frame, cropped_face, face_bbox = result
        x, y, w, h = face_bbox
        # Display the cropped face if available
        if cropped_face.size > 20000:
            reshaped_array = np.transpose(cropped_face, (2, 0, 1))
            reshaped_array = np.expand_dims(reshaped_array, axis=0)
            img_embedding = encode(reshaped_array)
            # print(img_embedding.shape)
            detect_dict = {}
            for k, v in all_people_faces.items():
                detect_dict[k] = (v - img_embedding).norm().item()
            min_key = min(detect_dict, key=detect_dict.get)
            # print(detect_dict)
            if detect_dict[min_key] >= thres:
                min_key = 'Undetected'

            cv2.rectangle(original_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(original_frame, (x, y+h - 35), (x+w, y+h), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(original_frame, min_key, (x + 6, y+h - 6), font, 1.5, (255, 255, 255), 2)

            # cv2.putText(
            #     original_frame, min_key, (x + 5, y + 10), 
            #     cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 3)

        
        # cv2.namedWindow("output", 0)
        # cv2.resizeWindow("output", 960, 640)
        cv2.imshow("output", original_frame)

        # Break the loop if the user hits the 'Esc' key
        if cv2.waitKey(1) & 0xFF == 27:
            break


    # end_time = time.time()
    # execution_time = end_time - start_time
    # Print the execution time
    # print(f"Execution time: {execution_time} seconds")
    camera_capture.close_camera()
