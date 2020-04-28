CC=cc

CFLAGS=-O3

LDFLAGS=

LIBS=-largp

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

c-sir: c-SIR.o
	$(CC) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	$(RM) -v *.o
