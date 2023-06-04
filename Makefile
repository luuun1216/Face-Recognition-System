# Variables
CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC $(shell python3 -m pybind11 --includes)
LDFLAGS = $(shell pkg-config --cflags --libs opencv4)
PYTHON_EXT_SUFFIX = $(shell python3-config --extension-suffix)
TARGET1 = camera_capture$(PYTHON_EXT_SUFFIX)
SRC1 = camera_capture.cpp


# Rules
all: $(TARGET1)

$(TARGET1): $(SRC1)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET1)

