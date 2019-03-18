CC=/usr/local/cuda-8.0/bin/nvcc
CFLAGS=-std=c++11 -O3 -ccbin /usr/bin/g++-4.8
LDFLAGS=-lgtest -lgtest_main -L/usr/lib -lpthread
INCLUDE=-I /usr/local/include/gtest -I ../include

all: test

test: main.cu ../include/*
	$(CC) main.cu -o test.out $(LDFLAGS) $(CFLAGS) $(INCLUDE)

clean:
	rm -rf *.o *~ *.h *.gcda *.gcno *.info test.out