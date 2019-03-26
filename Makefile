CC=/usr/local/cuda-8.0/bin/nvcc
CFLAGS=-std=c++11 -ccbin /usr/bin/g++-4.8
LDFLAGS=-lgtest -lgtest_main -L/usr/lib -lpthread
INCLUDE=-I /usr/local/include/gtest -I include

all: opt

opt: test/test.cu
	$(CC) test/test.cu -o bin/test.out -O3 $(LDFLAGS) $(CFLAGS) $(INCLUDE)

debug: test/test.cu
	$(CC) test/test.cu -o bin/test.out -O0 -g $(LDFLAGS) $(CFLAGS) $(INCLUDE)

clean:
	rm -rf bin/*