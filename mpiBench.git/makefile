# for linux
CC=mpicc

# for AIX
#CC=mpxlc

mpiBench : clean
	$(CC) $(CFLAGS) -o mpiBench mpiBench.c

check : clean
	$(CC) $(CFLAGS) -D CHECK_ALWAYS -o mpiBench mpiBench.c

nobar : clean
	$(CC) $(CFLAGS) -D NO_BARRIER -o mpiBench mpiBench.c

debug : clean
	$(CC) $(CFLAGS) -g -O0 -o mpiBench mpiBench.c

clean :
	rm -f mpiBench *.o
