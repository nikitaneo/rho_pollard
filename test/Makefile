CC=/usr/bin/g++-4.8
CFLAGS=-std=c++11 -O0 -g
LDFLAGS=--coverage -lgtest -lgtest_main -L/usr/lib -lpthread -Werror -Wall
INCLUDE=-I /usr/local/include/gtest -I ../include

all: test

test: main.cpp ../include/*
	$(CC) main.cpp -o test.out $(LDFLAGS) $(CFLAGS) $(INCLUDE)

clean:
	rm -rf *.o *~ *.h *.gcda *.gcno *.info test.out