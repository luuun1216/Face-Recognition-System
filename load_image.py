import cv2
import numpy as np
import image_processing


def load_image(image_path):
    return cv2.imread(image_path)

def main():
    '''
    # Test code - only show and save the pic 
    # input_image_path = '../pic.png'
    # output_image_path = '../image.jpg'
    # input_image = load_image(input_image_path)
    # processed_image = image_processing.process_image(input_image)
    # # cv2.imshow(processed_image)
    # cv2.imwrite(output_image_path, processed_image)
    # print(f"Processed image saved to {output_image_path}")
    '''
    # Open the camera (0 represents the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    try:
        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Process the frame using the C++ extension
            processed_frame = image_processing.process_image(frame)
            

            # Display the original and processed frames
            cv2.imshow('Original', frame)
            # cv2.imshow('Processed', processed_frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()