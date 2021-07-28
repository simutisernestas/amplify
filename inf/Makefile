
# CFLAGS=-g -Wall -Wextra `pkg-config --cflags --libs opencv4` -lonnxruntime
CFLAGS=-O3 -Wall -Wextra `pkg-config --cflags --libs opencv4` -lonnxruntime

build: main.cpp
	mkdir -p build
	g++ main.cpp -o build/inf $(CFLAGS)