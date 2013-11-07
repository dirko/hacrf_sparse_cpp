CC=g++
CFLAGS=-c -Wall -O3
LFLAGS=-Wall -Wno-unused -Wno-sign-compare -Wno-char-subscripts 
LDFLAGS=

learning:learning.o crf_ed.o 
	$(CC) $(LFLAGS) learning.o crf_ed.o -o learning -l lbfgs

learning.o:learning.cpp 
	$(CC) $(CFLAGS) learning.cpp -o learning.o -L/

crf_ed.o:crf_ed.cpp crf_ed.h
	$(CC) $(CFLAGS) crf_ed.cpp -o crf_ed.o

clean:
	rm learning *.o
