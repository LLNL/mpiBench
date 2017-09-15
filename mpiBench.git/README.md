# mpiBench
Times MPI collectives over a series of message sizes

# What is mpiBench?

mpiBench.c

This program measures MPI collective performance for a range of
message sizes.  The user may specify:
- the collective to perform,
- the message size limits,
- the number of iterations to perform,
- the maximum memory a process may allocate for MPI buffers,
- the maximum time permitted for a given test,
- and the number of Cartesian dimensions to divide processes into.

The default behavior of mpiBench will run from 0-256K byte messages
for all supported collectives on MPI_COMM_WORLD with a 1G buffer
limit.  Each test will execute as many iterations as it can to fit
within a default time limit of 50000 usecs.

crunch_mpiBench

This is a perl script which can be used to filter data and generate
reports from mpiBench output files.  It can merge data from
multiple mpiBench output files into a single report.  It can also
filter output to a subset of collectives.  By default, it reports
the operation duration time (i.e., how long the collective took to
complete).  For some collectives, it can also report the effective
bandwidth.  If provided two datasets, it computes a speedup factor.

# What is measured

mpiBench measures the total time required to iterate through a loop
of back-to-back invocations of the same collective (optionally
separated by a barrier), and divides by the number of iterations.
In other words the timing kernel looks like the following:

    time_start = timer();
    for (i=0 ; i < iterations; i++) {
      collective(msg_size);
      barrier();
    }
    time_end = timer();
    time = (time_end - time_start) / iterations;
 
Each participating MPI process performs this measurement and all
report their times.  It is the average, minimum, and maximum across
this set of times which is reported.

Before the timing kernel is started, the collective is invoked once to
prime it, since the initial call may be subject to overhead that later
calls are not.  Then, the collective is timed across a small set of
iterations (~5) to get a rough estimate for the time required for a
single invocation.  If the user specifies a time limit using the -t
option, this esitmate is used to reduce the number of iterations made
in the timing kernel loop, as necessary, so it may executed within the
time limit.


# Basic Usage

Build:

    make

  Run:

    srun -n <procs> ./mpiBench > output.txt

  Analyze:

    crunch_mpiBench output.txt

# Build Instructions

There are several make targets available:
- make       -- simple build
- make nobar -- build without barriers between consecutive collective invocations
- make debug -- build with "-g -O0" for debugging purposes
- make clean -- clean the build

If you'd like to build manually without the makefiles, there are some
compile-time options that you should be aware of:

  -D NO_BARRIER       - drop barrier between consecutive collective
                        invocations
  -D USE_GETTIMEOFDAY - use gettimeofday() instead of MPI_Wtime() for
                        timing info

# Usage Syntax

    Usage:  mpiBench [options] [operations]
  
    Options:
      -b <byte>  Beginning message size in bytes (default 0)
      -e <byte>  Ending message size in bytes (default 1K)
      -i <itrs>  Maximum number of iterations for a single test
                 (default 1000)
      -m <byte>  Process memory buffer limit (send+recv) in bytes
                 (default 1G)
      -t <usec>  Time limit for any single test in microseconds
                 (default 0 = infinity)
      -d <ndim>  Number of dimensions to split processes in
                 (default 0 = MPI_COMM_WORLD only)
      -c         Check receive buffer for expected data in last
                 interation (default disabled)
      -C         Check receive buffer for expected data every
                 iteration (default disabled)
      -h         Print this help screen and exit
      where <byte> = [0-9]+[KMG], e.g., 32K or 64M
  
    Operations:
      Barrier
      Bcast
      Alltoall, Alltoallv
      Allgather, Allgatherv
      Gather, Gatherv
      Scatter
      Allreduce
      Reduce

# Examples

## mpiBench

Run the default set of tests:

    srun -n2 -ppdebug mpiBench

Run the default message size range and iteration count for Alltoall, Allreduce, and Barrier:

    srun -n2 -ppdebug mpiBench Alltoall Allreduce Barrier

Run from 32-256 bytes and time across 100 iterations of Alltoall:

    srun -n2 -ppdebug mpiBench -b 32 -e 256 -i 100 Alltoall

Run from 0-2K bytes and default iteration count for Gather, but
reduce the iteration count, as necessary, so each message size
test finishes within 100,000 usecs:

    srun -n2 -ppdebug mpiBench -e 2K -t 100000 Gather

## crunch_mpiBench

Show data for just Alltoall:

    crunch_mpiBench -op Alltoall out.txt

Merge data from several files into a single report:

    crunch_mpiBench out1.txt out2.txt out3.txt

Display effective bandwidth for Allgather and Alltoall:

    crunch_mpiBench -bw -op Allgather,Alltoall out.txt

Compare times in output files in dir1 with those in dir2:

    crunch_mpiBench -data DIR1_DATA dir1/* -data DIR2_DATA dir2/*

# Additional Notes

Rank 0 always acts as the root process for collectives which involve
a root.

If the minimum and maximum are quite different, then some processes
may be escaping ahead to start later iterations before the last one
has completely finished.  In this case, one may use the maximum time
reported or insert a barrier between consecutive invocations (build
with "make" instead of "make nobar") to syncronize the processes.

For Reduce and Allreduce, vectors of doubles are added, so message
sizes of 1, 2, and 4-bytes are skipped.

Two available make commands build mpiBench with test kernels like
the following:

       "make"              "make nobar"
    start=timer()        start=timer()
    for(i=o;i<N;i++)     for(i=o;i<N;i++)
    {                    {
      MPI_Gather()         MPI_Gather()
      MPI_Barrier()
    }                    }
    end=timer()          end=timer()
    time=(end-start)/N   time=(end-start)/N

"make nobar" may allow processes to escape ahead, but does not
include cost of barrier.
