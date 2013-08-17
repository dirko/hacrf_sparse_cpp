CC=g++
CFLAGS=-c -Wall -O3
LFLAGS=-Wall -Wno-unused -Wno-sign-compare -Wno-char-subscripts 
LDFLAGS=

#optimization_ex:optimization_ex.cpp dlib/*.cpp
#	$(CC) $(LFLAGS) -Wno-unused -Wno-sign-compare -Wno-char-subscripts optimization_ex.cpp dlib/*.cpp -o optimization_ex

#extract_linear_features: extract_linear_features.o
#	$(CC) $(LFLAGS) extract_linear_features.o -o extract_linear_features

#extract_linear_features.o: extract_linear_features.cpp
#	$(CC) $(CFLAGS) extract_linear_features.cpp -o extract_linear_features.o

learning:learning.o crf_ed.o
	$(CC) $(LFLAGS) learning.o alglib/src/*.cpp crf_ed.o -o learning

learning.o:learning.cpp 
	$(CC) $(CFLAGS) learning.cpp -o learning.o -L/

crf_ed.o:crf_ed.cpp crf_ed.h
	$(CC) $(CFLAGS) crf_ed.cpp -o crf_ed.o

clean:
	rm learning *.o
