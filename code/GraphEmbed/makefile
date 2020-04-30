CC = g++
CFLAGS = -lm -pthread -Ofast -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas -std=c++11
INCLUDES = -I/home/liuy/Install/gsl-2.4/include -I./eigen
LIBS = -L/home/liuy/Install/gsl-2.4/lib


line : ransampl.o linelib.o main.o
	$(CC) -o hin2vec ransampl.o linelib.o main.o $(INCLUDES) $(LIBS) $(CFLAGS)

ransampl.o : ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS)

linelib.o : linelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c linelib.cpp $(INCLUDES) $(LIBS)

main.o : main.cpp linelib.o
	$(CC) $(CFLAGS) -c main.cpp $(INCLUDES) $(LIBS)

clean :
	rm -rf *.o hin2vec
