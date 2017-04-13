TARGET = eigen-getting-started
CFLAGS = -Wall -std=c++11
INCLUDES = \
	#-I../eigen-eigen-67e894c6cd8f

CC = g++
OBJ_DIR = ./obj

%.o: %.cpp
	@echo -n "Compiling:  $<\n"
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@ 

$(TARGET): $(TARGET).o
	@echo -n "Linking: $^\n"
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm *.o $(OBJ_DIR)/*.o
