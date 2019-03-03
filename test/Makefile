CC=/usr/bin/g++-4.8
CFLAGS=-std=c++11 -g -O0
LDFLAGS=-lgtest -lgtest_main -L/usr/lib -lpthread
INCLUDE=-I /usr/local/include/gtest -I ../include

all: test

test: main.cpp ../include/*
	$(CC) main.cpp -o test $(LDFLAGS) $(CFLAGS) $(INCLUDE)

clean:
	rm -rf *.o *~ *.h test