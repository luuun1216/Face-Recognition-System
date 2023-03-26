Face Recognition System
===========================
## Basic Information
Developing a Face Recognition System Using C++ and Python

**Github repository**: https://github.com/luuun1216/Face-Recognition-System

## Problem to Solve

Face recognition is a challenging task in computer vision that has numerous applications, such as security, surveillance, and human-computer interaction. Traditional methods for face recognition have limitations in terms of accuracy, efficiency, and scalability. Moreover, the emergence of deep learning has opened up new possibilities for improving the performance of face recognition systems, but it requires specialized knowledge and tools to build and train effective models. Therefore, the aim of this project is to develop a face recognition system that combines the strengths of C++ and Python and leverages deep learning to achieve high accuracy and efficiency, users only need to put the picture into the folder or open the camera, the system can automatic trainin or show the final result.

## Prospective Users

The face recognition system has numerous potential users, including security and surveillance companies, law enforcement agencies, and government organizations. The system can be used for applications such as access control, identity verification, and criminal investigation. Moreover, the system can also be used in human-computer interaction scenarios, such as user identification and personalized content delivery. 

## System Architecture
The deep learning model will be built using a convolutional neural network (CNN) architecture, which has been shown to be effective for face recognition tasks. The model will be trained using a large dataset of face images, such as the popular Labeled Faces in the Wild (LFW) dataset, and will use techniques such as data augmentation, regularization, and optimization to improve its performance.

Once the model is trained, the face recognition system will use it to perform the recognition task on new images. The system will first pre-process the input image using C++ functions to normalize the image, crop the face region, and extract features. The pre-processed image will then be passed to the Python code for classification using the trained deep learning model. Finally, the system will display the results, such as the detected face and the corresponding identity.


## API Description


## Engineering Infrastructure
### Automatic build system and how to build your program
- Use Conda or ip to install the dependent tools or library

### Version Control

- Git

### Programing Language

- Video split function, Camera function and image pre-processing function will be completed by C++.

- face reconition function and Generate the report function will be completed by Python.

### Make-Pybind-Pytest

- The build system will convert the c++ function into python function through make and pybind.

- The python code will be tested by pytest.


## Schedule

| Week | Schedule |                                                                                                     
| ------------- | ------------- |
| Week 1: Project Planning and Setup  | 1. Define project scope and requirements<br />2. Plan project schedule and milestones<br />3. Set up development environment (install C++, Python, Pybind11, and necessary libraries)<br /> |
| Week 2: Image Pre-Processing | 1. Implement image loading and pre-processing functions in C++<br />2. Test image pre-processing functions and optimize for performance<br />|
| Week 3: Feature Extraction | 1. Implement feature extraction functions in C++ <br />2. Test image pre-processing functions and optimize for performance<br />|
| Week 4: Deep Learning Model Training | 1. Develop deep learning model using Python and a deep learning framework (such as PyTorch or TensorFlow) |
| Week 5: Model Integration | 1. Use Pybind11 to integrate the C++ image pre-processing and feature extraction functions with the Python model training and inference code<br />2. Test integrated model |
| Week 6: Face Recognition System Implementation |1. Implement the complete face recognition system, including the C++ and Python code for image pre-processing, feature extraction, model inference, and result display |
| Week 7: System Evaluation and build workflow | 1. Evaluate the system performance and accuracy using benchmark datasets and metrics<br />2. Write Makefile or requirement.txt |
| Week 8: Final check and prepare presentation  | 1. Final check<br /> 2. Think about what I need to say |

## References
1. [Face Recognition] https://github.com/ageitgey/face_recognition
2. [Real-time Face recognition] https://github.com/s90210jacklen/Real-time-Face-recognition
3. [FaceNet: A Unified Embedding for Face Recognition and Clustering] https://arxiv.org/pdf/1503.03832.pdf
