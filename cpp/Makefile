CXX = g++
CXXFLAGS = -std=c++17 -I/home/hafiz/vcpkg/installed/x64-linux/include

# Targets
all: preprocess predict

# Compile the preprocess.cpp source file
preprocess: preprocess.cpp
	$(CXX) $(CXXFLAGS) preprocess.cpp -o preprocess

# Compile the predict.cpp source file
predict: predict.cpp
	$(CXX) $(CXXFLAGS) predict.cpp -o predict

# Clean the compiled files
clean:
	rm -f preprocess predict
