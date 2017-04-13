TARGET = eigen-getting-started
CFLAGS = -Wall -std=c++11
INCLUDES = -I../eigen-eigen-67e894c6cd8f

CC = g++
OBJ_DIR = ./obj

%.o: %.cpp
	@echo -n "Compiling:  $<\n"
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@ 

$(TARGET): $(TARGET).o
	@echo -n "Linking: $^\n"
	$(CC) -o $@ $^ $(CFLAGS)
	@echo "Link success: $@\n";

.PHONY: clean

clean:
	rm -f ./*.o 2> /dev/null
	rm -f $(OBJ_DIR)/*.o 2> /dev/null
