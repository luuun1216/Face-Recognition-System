# Face-Recognition-System

## Basic Information
Developing a Face Recognition System Using C++ and Python

**Github repository**: https://github.com/luuun1216/Face-Recognition-System

## Problem to Solve

Face recognition is a challenging task in computer vision that has numerous applications, such as security, surveillance, and human-computer interaction. Traditional methods for face recognition have limitations in terms of accuracy, efficiency, and scalability. Moreover, the emergence of deep learning has opened up new possibilities for improving the performance of face recognition systems, but it requires specialized knowledge and tools to build and train effective models. Therefore, the aim of this project is to develop a face recognition system that combines the strengths of C++ and Python and leverages deep learning to achieve high accuracy and efficiency.

## Methods Description

The face recognition system will be built using a combination of C++ and Python and will leverage the Pybind11 library to integrate the two languages. The C++ part of the system will be responsible for handling the image processing tasks, such as reading in images, pre-processing, and feature extraction. The Python part of the system will be responsible for training the deep learning model and performing the face recognition task.

The deep learning model will be built using a convolutional neural network (CNN) architecture, which has been shown to be effective for face recognition tasks. The model will be trained using a large dataset of face images, such as the popular Labeled Faces in the Wild (LFW) dataset, and will use techniques such as data augmentation, regularization, and optimization to improve its performance.

Once the model is trained, the face recognition system will use it to perform the recognition task on new images. The system will first pre-process the input image using C++ functions to normalize the image, crop the face region, and extract features. The pre-processed image will then be passed to the Python code for classification using the trained deep learning model. Finally, the system will display the results, such as the detected face and the corresponding identity.

## Prospective Users

The face recognition system has numerous potential users, including security and surveillance companies, law enforcement agencies, and government organizations. The system can be used for applications such as access control, identity verification, and criminal investigation. Moreover, the system can also be used in human-computer interaction scenarios, such as user identification and personalized content delivery. Overall, the face recognition system has the potential to offer high accuracy and efficiency in a variety of applications and can benefit users from various domains.

## System Architecture



## API Description

### Modules

There will have five modules on this software system, which is present as below.



### Example


## Engineering Infrastructure

### Version Control

- Git

### Programing Language

- Video split function, Camera function and image pre-processing function will be completed by c++.

- face reconition function and Generate the report function will be completed by python.

### Make-Pybind-Pytest

- The build system will convert the c++ function into python function through make and pybind.

- The python code will be tested by pytest.

### Project Step

Analysis report will be created using FPDF.

This project will be completed by executing the following steps:

- [ ] XXX

- [ ] XXX

- [ ] XXX

- [ ] Python binding.

- [ ] Final testing.

## Schedule

| Week | Schedule |                                                                                                     
| ------------- | ------------- |
| Week 1  |OOO|
| Week 2  | 1. OOO<br />2. XXX<br />3. OOO<br /> |
| Week 3  | OOO | 
| Week 3  | OOO |
| Week 4  | OOO |
| Week 5  | OOO  |
| Week 6  | Testing function   |
| Week 7  | Build Workflow|
| Week 8  | Project Presentation  |

## References

