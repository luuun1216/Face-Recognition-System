#include <iostream>
#include <stdio.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 

using namespace std;
using namespace cv;
namespace py = pybind11;

cv::Mat process_image(cv::Mat &input_image) {
    cv::Mat faces_images;
    cv::Mat frame_gray;
    std::vector<Rect> faces;
    String face_cascade_name = "../source/haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    /* Load cascade classifiers */
	if(!face_cascade.load(face_cascade_name))
		cout << "Error loading face cascade";

    /* test code - only for BGR2gray */
    // Perform your image processing here, for example, convert to grayscale
    // cv::cvtColor(input_image, output_image, cv::COLOR_BGR2GRAY);
    
    cv::cvtColor(input_image, frame_gray, COLOR_BGR2GRAY);

    cv::equalizeHist(frame_gray, frame_gray);
    
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 5, CASCADE_SCALE_IMAGE, Size(30, 30));
    for (size_t i = 0; i < faces.size(); i++)
    {
        /* Draw rectangular on face */
        rectangle(input_image, faces[i], Scalar(255, 0, 0), 3, 8, 0);
        
        // cv::Mat faceROI = frame_gray(faces[i]);
        // /* Add face ROI to vector */
        // faces_images.push_back(faceROI);
    }

    return input_image;
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

py::array_t<unsigned char> cv_mat_to_numpy_uint8_3c(cv::Mat &mat) {
    return py::array_t<unsigned char>({mat.rows, mat.cols, 3}, mat.data);
}

py::array_t<unsigned char> cv_mat_to_numpy_uint8_1c(cv::Mat &mat) {
    return py::array_t<unsigned char>({mat.rows, mat.cols}, mat.data);
}

py::array_t<unsigned char> process_image_pybind(py::array_t<unsigned char> &input_array) {
    cv::Mat input_image = numpy_uint8_3c_to_cv_mat(input_array);
    cv::Mat output_image = process_image(input_image);
    return cv_mat_to_numpy_uint8_1c(output_image);
}

PYBIND11_MODULE(image_processing, m) {
    m.def("process_image", &process_image_pybind, "Process the image using OpenCV");
}