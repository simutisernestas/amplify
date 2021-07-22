
CFLAGS=-g -Wall -Wextra `pkg-config --cflags --libs opencv4` -lonnxruntime

build: main.cpp
	g++ main.cpp -o inf $(CFLAGS)
	./inf