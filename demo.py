import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from types import MethodType
import image_processing
import numpy as np
# helper function
def encode(img):
    # print(type(img))
    res = resnet(torch.Tensor(img))
    return res
# load model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# get encoded features for all saved images
saved_pictures = "./save"
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

def detect(cam=0, thres=0.5):
    cap = cv2.VideoCapture(cam)
    try:
        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Process the frame using the C++ extension
            result = image_processing.process_image(frame)
            original_frame, cropped_face, face_bbox = result
            x, y, w, h = face_bbox

            cv2.imshow('Original', original_frame)
            # print(f"Bounding box coordinates: x={x}, y={y}, width={w}, height={h}")

            # Display the cropped face if available
            if cropped_face.size > 0:
                cv2.imshow('Cropped Face', cropped_face)
                # print(cropped_face.shape)
                reshaped_array = np.transpose(cropped_face, (2, 0, 1))
                # print(reshaped_array.shape)
                reshaped_array = np.expand_dims(reshaped_array, axis=0)
                
                img_embedding = encode(reshaped_array)
                # print(img_embedding)
                detect_dict = {}
                for k, v in all_people_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()
                min_key = min(detect_dict, key=detect_dict.get)
                # print(min_key)
                # print(detect_dict[min_key])
                if detect_dict[min_key] >= thres:
                    min_key = 'Undetected'

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(
                frame, min_key, (x + 5, y + 10), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("output", frame)
            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect(cam=0)