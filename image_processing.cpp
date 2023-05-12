#include <iostream>
#include <stdio.h>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 

using namespace std;
using namespace cv;
namespace py = pybind11;

std::tuple<cv::Mat, cv::Mat, cv::Rect> process_image(cv::Mat &input_image) {
    cv::Mat frame_gray, cropped_face;
    cv::Rect largest_face_bbox;
    std::vector<cv::Rect> faces;
    std::string face_cascade_name = "../source/haarcascade_frontalface_default.xml";
    cv::CascadeClassifier face_cascade;

    // Load cascade classifiers
    if (!face_cascade.load(face_cascade_name)) {
        std::cout << "Error loading face cascade" << std::endl;
    }

    cv::cvtColor(input_image, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 5, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    if (!faces.empty()) {
        // Find the largest face
        auto largest_face = std::max_element(faces.begin(), faces.end(),
                                             [](const cv::Rect &a, const cv::Rect &b) {
                                                 return a.area() < b.area();
                                             });

        // Draw a rectangle on the largest face
        // cv::rectangle(input_image, *largest_face, cv::Scalar(255, 0, 0), 3, 8, 0);

        // Crop the largest face
        cropped_face = input_image(*largest_face);

        // Save the largest face bounding box
        largest_face_bbox = *largest_face;
    }

    return std::make_tuple(input_image, cropped_face, largest_face_bbox);
}

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char> &input_array) {
    py::buffer_info buf = input_array.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
    return mat;
}

cv::Mat numpy_uint8_1c_to_cv_mat(py::array_t<unsigned char> &input_array) {
    py::buffer_info buf = input_array.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    return mat;
}

py::array_t<unsigned char> cv_mat_to_numpy_uint8_3c(const cv::Mat &mat) {
    // return py::array_t<unsigned char>({mat.rows, mat.cols, 3}, mat.data);
    return py::array_t<unsigned char>({mat.rows, mat.cols, 3}, {mat.step[0], mat.step[1], sizeof(unsigned char)}, mat.data);
}

py::array_t<unsigned char> cv_mat_to_numpy_uint8_1c(cv::Mat &mat) {
    return py::array_t<unsigned char>({mat.rows, mat.cols}, mat.data);
}

py::tuple process_image_pybind(py::array_t<unsigned char> &input_array) {
    cv::Mat input_image = numpy_uint8_3c_to_cv_mat(input_array);
    cv::Mat original_image, cropped_face;
    cv::Rect face_bbox;
    std::tie(original_image, cropped_face, face_bbox) = process_image(input_image);

    return py::make_tuple(cv_mat_to_numpy_uint8_3c(original_image),
                          cv_mat_to_numpy_uint8_3c(cropped_face),
                          py::make_tuple(face_bbox.x, face_bbox.y, face_bbox.width, face_bbox.height));
}

PYBIND11_MODULE(image_processing, m) {
    m.def("process_image", &process_image_pybind, "Process the image using OpenCV");
}