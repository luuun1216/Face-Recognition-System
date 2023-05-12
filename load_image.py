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
            result = image_processing.process_image(frame)
            original_frame_with_rectangles, cropped_face, face_bbox = result
            x, y, w, h = face_bbox
            
            print(f"Bounding box coordinates: x={x}, y={y}, width={w}, height={h}")

            # Display the original frame with rectangles
            cv2.imshow('Original', original_frame_with_rectangles)

            # Display the cropped face if available
            if cropped_face.size > 0:
                cv2.imshow('Cropped Face', cropped_face)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()