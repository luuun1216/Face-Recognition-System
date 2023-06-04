#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

cv::VideoCapture cap;

bool open_input_source(std::string source) {
    // If source can be converted to an integer, assume it's a camera ID
    try {
        int cameraID = std::stoi(source);
        cap.open(cameraID, cv::CAP_ANY);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    } catch (const std::invalid_argument&) {
        // If source can't be converted to an integer, assume it's a video file path
        cap.open(source);
    }

    if (!cap.isOpened()) {
        std::cout << "ERROR! Unable to open camera or video file\n";
        return false;
    }
    return true;
}

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char> &input_array) {
    py::buffer_info buf_info = input_array.request();
    cv::Mat mat(buf_info.shape[0], buf_info.shape[1], CV_8UC3, (unsigned char*)buf_info.ptr);
    return mat;
}

py::array_t<unsigned char> cv_mat_to_numpy_uint8_3c(const cv::Mat &mat) {
    return py::array_t<unsigned char>({mat.rows, mat.cols, 3}, {mat.step[0], mat.step[1], sizeof(unsigned char)}, mat.data);
}

py::array_t<unsigned char> capture_frame() {
    cv::Mat frame;
    cap.read(frame);

    return cv_mat_to_numpy_uint8_3c(frame);
}

std::vector<cv::Rect> detect_faces(cv::Mat &frame) {
    cv::Mat frame_gray;
    std::vector<cv::Rect> faces;
    cv::CascadeClassifier face_cascade;
    std::string face_cascade_name = "../source/haarcascade_frontalface_default.xml";

    // Load cascade classifier
    if (!face_cascade.load(face_cascade_name)) {
        std::cout << "Error loading face cascade";
        return faces;
    }

    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 5, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    return faces;
}

py::tuple rect_to_tuple(const cv::Rect& rect) {
    return py::make_tuple(rect.x, rect.y, rect.width, rect.height);
}

py::tuple process_image_pybind() {
    cv::Mat frame;
    cap.read(frame);

    // Perform face detection
    std::vector<cv::Rect> faces = detect_faces(frame);
    cv::Rect largest_face_bbox;
    cv::Mat cropped_face;

    if (!faces.empty()) {
        // Find the largest face
        auto largest_face = std::max_element(faces.begin(), faces.end(),
                                             [](const cv::Rect &a, const cv::Rect &b) {
                                                 return a.area() < b.area();
                                             });
        // Crop the largest face
        cropped_face = frame(*largest_face);

        // Save the largest face bounding box
        largest_face_bbox = *largest_face;
    }

    // Return the frame, cropped face and bounding box as a tuple
    return py::make_tuple(cv_mat_to_numpy_uint8_3c(frame), 
                          cv_mat_to_numpy_uint8_3c(cropped_face), 
                          rect_to_tuple(largest_face_bbox));
}


void close_camera() {
    if (cap.isOpened()) {
        cap.release();
    }
}

PYBIND11_MODULE(camera_capture, m) {
    m.def("open_input_source", &open_input_source, "Open the camera or video file");
    m.def("capture_frame", &capture_frame, "Capture a frame from the camera");
    m.def("process_image", &process_image_pybind, "Process a frame from the camera");
    m.def("close_camera", &close_camera, "Close the camera");
}