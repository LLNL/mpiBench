/*
Copyright (c) 2007-2008, Lawrence Livermore National Security (LLNS), LLC
Produced at the Lawrence Livermore National Laboratory (LLNL)
Written by Adam Moody <moody20@llnl.gov>.
UCRL-CODE-232117.
All rights reserved.

This file is part of mpiGraph. For details, see
  http://www.sourceforge.net/projects/mpigraph
Please also read the Additional BSD Notice below.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
â* Redistributions of source code must retain the above copyright notice, this
   list of conditions and the disclaimer below.
â* Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the disclaimer (as noted below) in the documentation
   and/or other materials provided with the distribution.
â* Neither the name of the LLNL nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific prior
   written permission.
â* 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL LLNL, THE U.S. DEPARTMENT
OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Additional BSD Notice
1. This notice is required to be provided under our contract with the U.S. Department
   of Energy (DOE). This work was produced at LLNL under Contract No. W-7405-ENG-48
   with the DOE.
2. Neither the United States Government nor LLNL nor any of their employees, makes
   any warranty, express or implied, or assumes any liability or responsibility for
   the accuracy, completeness, or usefulness of any information, apparatus, product,
   or process disclosed, or represents that its use would not infringe privately-owned
   rights.
3. Also, reference herein to any specific commercial products, process, or services
   by trade name, trademark, manufacturer or otherwise does not necessarily constitute
   or imply its endorsement, recommendation, or favoring by the United States Government
   or LLNL. The views and opinions of authors expressed herein do not necessarily state
   or reflect those of the United States Government or LLNL and shall not be used for
   advertising or product endorsement purposes.
*/

/* 
Comiple flags:
  -DNO_BARRIER       - Drops MPI_Barrier() call that separates consecutive collective calls
  -DUSE_GETTIMEOFDAY - Use gettimeofday() for timing rather than MPI_WTime()
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <signal.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#if 0
#ifndef _AIX
#include "print_mpi_resources.h"
#endif
#endif

char VERS[] = "1.5";

/*
------------------------------------------------------
Globals
------------------------------------------------------
*/
#define KILO (1024)
#define MEGA (KILO*KILO)
#define GIGA (KILO*MEGA)

#define ITRS_EST       (5)        /* Number of iterations used to estimate time */
#define ITRS_RUN       (1000)     /* Number of iterations to run (without timelimits) */
#define MSG_SIZE_START (0)        /* Lower bound of message sizes in bytes */
#define MSG_SIZE_STOP  (256*KILO) /* Upper bound of message sizes in bytes */
#define MAX_PROC_MEM   (1*GIGA)   /* Limit on MPI buffer sizes in bytes */

/* Compile with -DNO_BARRIER to drop barriers between collective calls
     Adds barrier between test iterations to sync all procs before issuing next collective
     Prevents non-root MPI ranks from escaping ahead into future iterations
     Barrier overhead is not subtracted from timing results
*/
#ifdef NO_BARRIER
  #define __BAR__(comm)
#else
  #define __BAR__(comm) MPI_Barrier(comm)
#endif

/* we use a bit mask to flag which collectives to test */
#define BARRIER    (0x001)
#define BCAST      (0x002)
#define ALLTOALL   (0x004)
#define ALLGATHER  (0x008)
#define GATHER     (0x010)
#define SCATTER    (0x020)
#define ALLREDUCE  (0x040)
#define REDUCE     (0x080)
#define ALLTOALLV  (0x100)
#define ALLGATHERV (0x200)
#define GATHERV    (0x400)
#define NUM_TESTS  (11)

char* TEST_NAMES[] = {
  "Barrier", "Bcast", "Alltoall", "Allgather", "Gather", "Scatter", "Allreduce", "Reduce", "Alltoallv", "Allgatherv", "Gatherv"
};
int   TEST_FLAGS[] = {
   BARRIER,   BCAST,   ALLTOALL,   ALLGATHER,   GATHER,   SCATTER,   ALLREDUCE,   REDUCE,   ALLTOALLV,   ALLGATHERV,   GATHERV
};
  
int rank_local; /* my MPI rank */
int rank_count; /* number of ranks in job */
int dimid_key;
size_t allocated_memory = 0; /* number of bytes allocated */

/*
------------------------------------------------------
Utility Functions
------------------------------------------------------
*/

/* Print usage syntax and exit */
int usage()
{
    if (rank_local == 0) {
        printf("\n");
        printf("  Usage:  mpiBench [options] [operations]\n");
        printf("\n");
        printf("  Options:\n");
        printf("    -b <byte>  Beginning message size in bytes (default 0)\n");
        printf("    -e <byte>  Ending message size in bytes (default 1K)\n");
        printf("    -m <byte>  Process memory buffer limit (send+recv) in bytes (default 1G)\n");
        printf("    -i <itrs>  Maximum number of iterations for a single test (default 1000)\n");
        printf("    -t <usec>  Time limit for any single test in microseconds (default 0 = infinity)\n");
        printf("    -d <ndim>  Number of Cartesian dimensions to split processes in (default 0 = MPI_COMM_WORLD only)\n");
        printf("    -p <size>  Minimum partition size (number of ranks) to divide MPI_COMM_WORLD by\n");
        printf("    -c         Check receive buffer for expected data in last interation (default disabled)\n");
        printf("    -C         Check receive buffer for expected data every iteration (default disabled)\n");
        printf("    -h         Print this help screen and exit\n");
        printf("    where <byte> = [0-9]+[KMG], e.g., 32K or 64M\n");
        printf("\n");
        printf("  Operations:\n");
        printf("    Barrier\n");
        printf("    Bcast\n");
        printf("    Alltoall, Alltoallv\n");
        printf("    Allgather, Allgatherv\n");
        printf("    Gather, Gatherv\n");
        printf("    Scatter\n");
        printf("    Allreduce\n");
        printf("    Reduce\n");
        printf("\n");
    }
    exit(1);
}

/* Allocate size bytes and keep track of amount allocated */
void* _ALLOC_MAIN_ (size_t size, char* debug) 
{
    void* p_buf;
    p_buf = malloc(size);
    if (!p_buf) {
        printf("ERROR:  Allocating memory %s:  requesting %ld bytes\n", debug, size);
        exit(1);
    }
    memset(p_buf, 0, size);
    allocated_memory += size;
    return p_buf;
}

/* Processes byte strings in the following format:
     <float_num>[kKmMgG][bB]
   and returns number of bytes as an size_t
   returns 0 on error
   Examples: 1K, 2.5kb, .5GB
*/
size_t atobytes(char* str)
{
    char* next;
    size_t units = 1;

    double num = strtod(str, &next);
    if (num == 0.0 && next == str) return 0;
    if (*next != 0) {
        /* process units for kilo, mega, or gigabytes */
        switch(*next) {
            case 'k':
            case 'K':
                units = (size_t) KILO;
                break;
            case 'm':
            case 'M':
                units = (size_t) MEGA;
                break;
            case 'g':
            case 'G':
                units = (size_t) GIGA;
                break;
            default:
                printf("ERROR:  unexpected byte string %s\n", str);
                exit(1);
        }
        next++;
        if (*next == 'b' || *next == 'B') { next++; } /* handle optional b or B character, e.g. in 10KB */
        if (*next != 0) {
            printf("ERROR:  unexpected byte string: %s\n", str);
            exit(1);
        }
    }
    if (num < 0) { printf("ERROR:  byte string must be positive: %s\n", str);  exit(1); }
    return (size_t) (num * (double) units);
}

/*
------------------------------------------------------
TIMING CODE - start/stop the timer and measure the difference
------------------------------------------------------
*/

#ifdef USE_GETTIMEOFDAY

/* use gettimeofday() for timers */
#include <sys/time.h>
#define __TIME_START__    (gettimeofday(&g_timeval__start, &g_timezone))
#define __TIME_END__      (gettimeofday(&g_timeval__end  , &g_timezone))
#define __TIME_USECS__    (d_Time_Diff_Micros(g_timeval__start, g_timeval__end))
#define d_Time_Diff_Micros(timeval__start, timeval__end) \
  ( \
    (double) (  (timeval__end.tv_sec  - timeval__start.tv_sec ) * 1000000 \
              + (timeval__end.tv_usec - timeval__start.tv_usec)  ) \
  )
#define d_Time_Micros(timeval) \
  ( \
    (double) (  timeval.tv_sec * 1000000 \
              + timeval.tv_usec  ) \
  )
struct timeval  g_timeval__start, g_timeval__end;
struct timezone g_timezone;

#else

/* use MPI_Wtime for timers (recommened)
   on some systems gettimeofday may be reset backwards by a global clock,
   which can even lead to negative length time intervals
*/
#define __TIME_START__    (g_timeval__start    = MPI_Wtime())
#define __TIME_END__      (g_timeval__end      = MPI_Wtime())
#define __TIME_USECS__    ((g_timeval__end - g_timeval__start) * 1000000.0)
double g_timeval__start, g_timeval__end;

#endif /* of USE_GETTIMEOFDAY */

/* Gather value from each task and print statistics */
double Print_Timings(double value, char* title, size_t bytes, int iters, MPI_Comm comm)
{
    int i;
    double min, max, avg, dev;
    double* times = NULL;

    if(rank_local == 0) {
        times = (double*) malloc(sizeof(double) * rank_count);
    }

    /* gather single time value from each task to rank 0 */
    MPI_Gather(&value, 1, MPI_DOUBLE, times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* rank 0 computes the min, max, and average over the set */
    if(rank_local == 0) {
        avg = 0;
        dev = 0;
        min = 100000000;
        max = -1;
        for(i = 0; i < rank_count; i++) {
            if(times[i] < min) { min = times[i]; }
            if(times[i] > max) { max = times[i]; }
            avg += times[i];
            dev += times[i] * times[i];
        }
        avg /= (double) rank_count;
        dev = 0; /*sqrt((dev / (double) rank_count - avg * avg)); */

        /* determine who we are in this communicator */
        int nranks, flag;
        char* str;
        MPI_Comm_get_attr(comm, dimid_key, (void*) &str, &flag); 
        MPI_Comm_size(comm, &nranks);

        printf("%-20.20s\t", title);
        printf("Bytes:\t%8u\tIters:\t%7d\t", bytes, iters);
        printf("Avg:\t%8.4f\tMin:\t%8.4f\tMax:\t%8.4f\t", avg, min, max);
        printf("Comm: %s\tRanks: %d\n", str, nranks);
        fflush(stdout);

        free((void*) times);
    }

    /* broadcast the average value back out */
    MPI_Bcast(&avg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return avg;
}

/*
------------------------------------------------------
MAIN
------------------------------------------------------
*/

char *sbuffer;
char *rbuffer;
int  *sendcounts, *sdispls, *recvcounts, *rdispls;
size_t buffer_size = 0;
int check_once;
int check_every;

struct argList {
    int    iters;
    size_t messStart;
    size_t messStop;
    size_t memLimit;
    double timeLimit;
    int    testFlags;
    int    checkOnce;
    int    checkEvery;
    int    ndims;
    int    partSize;
};

int processArgs(int argc, char **argv, struct argList* args)
{
  int i, j;
  char *argptr;
  char flag;

  /* set to default values */
  args->iters      = ITRS_RUN;
  args->messStart  = (size_t) MSG_SIZE_START;
  args->messStop   = (size_t) MSG_SIZE_STOP;
  args->memLimit   = (size_t) MAX_PROC_MEM;
  args->timeLimit  = 0;
  args->testFlags  = 0x1FFF;
  args->checkOnce  = 0;
  args->checkEvery = 0;
  args->ndims      = 0;
  args->partSize   = 0;

  int iters_set = 0;
  int time_set  = 0;
  for (i=0; i<argc; i++)
  {
    /* check for options */
    if (argv[i][0] == '-')
    {
      /* flag is the first char following the '-' */
      flag   = argv[i][1];
      argptr = NULL;

      /* single argument parameters */
      if (strchr("cC", flag))
      {
        switch(flag)
        {
        case 'c':
          args->checkOnce = 1;
          break;
        case 'C':
          args->checkEvery = 1;
          break;
        }
        continue;
      }
      
      /* check that we've got a valid option */
      if (!strchr("beithmdp", flag))
      {
        printf("\nInvalid flag -%c\n", flag);
        return(0);
      }
      
      /* handles "-i#" or "-i #" */
      if (argv[i][2] != 0) {
        argptr = &(argv[i][2]);
      } else {
        argptr = argv[i+1];
        i++;
      }

      switch(flag)
      {
      case 'b':
        args->messStart = atobytes(argptr);
        break;
      case 'e':
        args->messStop = atobytes(argptr);
        break;
      case 'i':
        args->iters = atoi(argptr);
        iters_set = 1;
        break;
      case 'm':
	args->memLimit = atobytes(argptr);
        break;
      case 't':
        args->timeLimit = (double) atol(argptr);
        time_set = 1;
        break;
      case 'd':
        args->ndims = atoi(argptr);
        break;
      case 'p':
        args->partSize = atoi(argptr);
        break;
      default:
        return(0);
      }
    }
    
    /* if the user gave no iteration limit and no time limit, set a reasonable time limit */
    if (!iters_set && !time_set) { args->timeLimit = 50000; }

    /* turn on test flags requested by user
       if user doesn't specify any, all will be run */
    for(j=0; j<NUM_TESTS; j++) {
      if(!strcasecmp(TEST_NAMES[j], argv[i])) {
        if(args->testFlags == 0x1FFF) args->testFlags = 0;
        args->testFlags |= TEST_FLAGS[j];
      }
    }
  }

  if (args->iters == 0)
  {
    printf("\n  Must define number of operations per measurement!\n\n");
    return(0);
  }

  return(1);
}

/* fill the send buffer with a known pattern */
void init_sbuffer(int rank)
{
    size_t i;
    char value;
    for(i=0; i<buffer_size; i++) {
        value = (char) ((i+1)*(rank+1) + i);
        sbuffer[i] = value;
    }
}

/* fill the receive buffer with a known pattern */
void init_rbuffer(int rank)
{
    /* nothing fancy here -- just blank it out */
    memset(rbuffer, 0, buffer_size);
}

/* check the send buffer for any deviation from expected pattern */
void check_sbuffer(int rank)
{
    size_t i;
    char value;
    for(i=0; i<buffer_size; i++) {
        value = (char) ((i+1)*(rank+1) + i);
        if (sbuffer[i] != value) {
            printf("Send buffer corruption detected on rank %d at sbuffer[%d]\n", rank, i);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

/* check the receive buffer for any deviation from expected pattern */
void check_rbuffer(char* buffer, size_t byte_offset, int rank, size_t src_byte_offset, size_t element_count)
{
    size_t i, j;
    char value;
    buffer += byte_offset;
    for(i=0, j=src_byte_offset; i<element_count; i++, j++) {
        value = (char) ((j+1)*(rank+1) + j);
        if (buffer[i] != value) {
              printf("Receive buffer corruption detected on rank %d at rbuffer[%d] from rank %d\n", rank_local, byte_offset+i, rank);
              MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

struct collParams {
    size_t   size;     /* message (element) size in bytes */
    int      iter;     /* number of iterations to test with */
    int      root;     /* root of collective operation */
    MPI_Comm comm;     /* communicator to test collective on */
    int      myrank;   /* my rank in the above communicator */
    int      nranks;   /* number of ranks in the above communicator */
    int      count;    /* element count for collective */
    MPI_Datatype type; /* MPI_Datatype to be used in collective (assumed contiguous) */
    MPI_Op   reduceop; /* MPI_Reduce operation to be used */
};

double time_barrier(struct collParams* p)
{
    int i;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        MPI_Barrier(p->comm);
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_bcast(struct collParams* p)
{
    int i;
    char* buffer = (p->myrank == p->root) ? sbuffer : rbuffer;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        int check = (check_every || (check_once && i == p->iter-1));
        if (check) {
          init_sbuffer(p->myrank);
          init_rbuffer(p->myrank);
        }

        MPI_Bcast(buffer, p->count, p->type, p->root, p->comm);
        __BAR__(p->comm);

        if (check) {
            check_sbuffer(p->myrank);
            check_rbuffer(buffer, 0, p->root, 0, p->size);
        }
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_alltoall(struct collParams* p)
{
    int i, j;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        int check = (check_every || (check_once && i == p->iter-1));
        if (check) {
            init_sbuffer(p->myrank);
            init_rbuffer(p->myrank);
        }

        MPI_Alltoall(sbuffer, p->count, p->type, rbuffer, p->size, p->type, p->comm);
        __BAR__(p->comm);

        if (check) {
            check_sbuffer(p->myrank);
            for (j = 0; j < p->nranks; j++) {
                check_rbuffer(rbuffer, j*p->size, j, p->myrank*p->size, p->size);
            }
        }
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_alltoallv(struct collParams* p)
{
    int i, j, k, count;
    int disp = 0;
    int chunksize = p->count / p->nranks;
    if (chunksize == 0) { chunksize = 1; }
    for (i = 0; i < p->nranks; i++) {
        int count = ((i+p->myrank)*chunksize) % (p->count+1);
        sendcounts[i] = count;
        recvcounts[i] = count;
        sdispls[i] = disp;
        rdispls[i] = disp;
        disp += count;
    }
    size_t scale = (p->count > 0) ? (p->size/p->count) : 0;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        int check = (check_every || (check_once && i == p->iter-1));
        if (check) {
            init_sbuffer(p->myrank);
            init_rbuffer(p->myrank);
        }

        MPI_Alltoallv(sbuffer, sendcounts, sdispls, p->type, rbuffer, recvcounts, rdispls, p->type, p->comm);
        __BAR__(p->comm);

        if (check) {
            check_sbuffer(p->myrank);
            for (k = 0; k < p->nranks; k++) {
                disp = 0;
                for (j = 0; j < p->myrank; j++) { disp += ((j+k)*chunksize) % (p->size+1); }
                check_rbuffer(rbuffer, rdispls[k]*scale, k, disp, recvcounts[k]*scale);
            }
        }
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_allgather(struct collParams* p)
{
    int i, j;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        int check = (check_every || (check_once && i == p->iter-1));
        if (check) {
            init_sbuffer(p->myrank);
            init_rbuffer(p->myrank);
        }

        MPI_Allgather(sbuffer, p->count, p->type, rbuffer, p->count, p->type, p->comm);
        __BAR__(p->comm);

        if (check) {
            check_sbuffer(p->myrank);
            for (j = 0; j < p->nranks; j++) {
                check_rbuffer(rbuffer, j*p->size, j, 0, p->size);
            }
        }
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_allgatherv(struct collParams* p)
{
    int i, j, count;
    int disp = 0;
    int chunksize = p->count / p->nranks;
    if (chunksize == 0) { chunksize = 1; }
    for ( i = 0; i < p->nranks; i++) {
        int count = (i*chunksize) % (p->count+1);
        recvcounts[i] = count;
        rdispls[i] = disp;
        disp += count;
    }
    size_t scale = (p->count > 0) ? (p->size/p->count) : 0;
    MPI_Barrier(MPI_COMM_WORLD);

    count = (p->myrank*chunksize) % (p->count+1);
    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        int check = (check_every || (check_once && i == p->iter-1));
        if (check) {
            init_sbuffer(p->myrank);
            init_rbuffer(p->myrank);
        }

        MPI_Allgatherv(sbuffer, count, p->type, rbuffer, recvcounts, rdispls, p->type, p->comm);
        __BAR__(p->comm);

        if (check) {
            check_sbuffer(p->myrank);
            for (j = 0; j < p->nranks; j++) {
                check_rbuffer(rbuffer, rdispls[j]*scale, j, 0, recvcounts[j]*scale);
            }
        }
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_gather(struct collParams* p)
{
    int i, j;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        int check = (check_every || (check_once && i == p->iter-1));
        if (check) {
            init_sbuffer(p->myrank);
            init_rbuffer(p->myrank);
        }

        MPI_Gather(sbuffer, p->count, p->type, rbuffer, p->count, p->type, p->root, p->comm);
        __BAR__(p->comm);

        if (check) {
            check_sbuffer(p->myrank);
            if (p->myrank == p->root) {
                for (j = 0; j < p->nranks; j++) {
                    check_rbuffer(rbuffer, j*p->size, j, 0, p->size);
                }
            }
        }
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_gatherv(struct collParams* p)
{
    int i, j, count;
    int disp = 0;
    int chunksize = p->count / p->nranks;
    if (chunksize == 0) { chunksize = 1; }
    for ( i = 0; i < p->nranks; i++) {
        int count = (i*chunksize) % (p->count+1);
        recvcounts[i] = count;
        rdispls[i] = disp;
        disp += count;
    }
    size_t scale = (p->count > 0) ? (p->size/p->count) : 0;
    MPI_Barrier(MPI_COMM_WORLD);

    count = (p->myrank*chunksize) % (p->count+1);
    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        int check = (check_every || (check_once && i == p->iter-1));
        if (check) {
            init_sbuffer(p->myrank);
            init_rbuffer(p->myrank);
        }

        MPI_Gatherv(sbuffer, count, p->type, rbuffer, recvcounts, rdispls, p->type, p->root, p->comm);
        __BAR__(p->comm);

        if (check) {
            check_sbuffer(p->myrank);
            if (p->myrank == p->root) {
                for (j = 0; j < p->nranks; j++) {
                    check_rbuffer(rbuffer, rdispls[j]*scale, j, 0, recvcounts[j]*scale);
                }
            }
        }
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_scatter(struct collParams* p)
{
    int i;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        int check = (check_every || (check_once && i == p->iter-1));
        if (check) {
            init_sbuffer(p->myrank);
            init_rbuffer(p->myrank);
        }

        MPI_Scatter(sbuffer, p->count, p->type, rbuffer, p->count, p->type, p->root, p->comm);
        __BAR__(p->comm);

        if (check) {
            check_sbuffer(p->myrank);
            check_rbuffer(rbuffer, 0, p->root, p->myrank*p->size, p->size);
        }
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_allreduce(struct collParams* p)
{
    int i;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        MPI_Allreduce(sbuffer, rbuffer, p->count, p->type, p->reduceop, p->comm);
        __BAR__(p->comm);
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

double time_reduce(struct collParams* p)
{
    int i;
    MPI_Barrier(MPI_COMM_WORLD);

    __TIME_START__;
    for (i = 0; i < p->iter; i++) {
        MPI_Reduce(sbuffer, rbuffer, p->count, p->type, p->reduceop, p->root, p->comm);
        __BAR__(p->comm);
    }
    __TIME_END__;

    return __TIME_USECS__ / (double)p->iter;
}

/* Prime, estimate, and time the collective called by the specified function
   for the given message size, iteration count, and time limit.  Then, print
   out the results.
*/
double get_time(double (*fn)(struct collParams* p), char* title, struct collParams* p, int iter, int time_limit)
{
    double time;
    double time_avg;
    int iter_limit;

    /* initialize the send and receive buffer with something */
    init_sbuffer(p->myrank);
    init_rbuffer(p->myrank);

    /* prime the collective with an intial call */
    p->iter = 1;
    time = fn(p);

    /* run through a small number of iterations to get a rough estimate of time */
    p->iter = ITRS_EST;
    time = fn(p);

    /* if a time limit has been specified, use the esitmate to limit the maximum number of iterations */
    iter_limit = (time_limit > 0   ) ? (int) (time_limit / time) : iter;
    iter_limit = (iter_limit < iter) ? iter_limit : iter;

    /* use the number calculated by the root (rank 0) which should be the slowest */
    MPI_Bcast(&iter_limit, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* run the tests (unless the limited iteration count is smaller than that used in the estimate) */
    if(iter_limit > ITRS_EST) {
        p->iter = iter_limit;
        time = fn(p);
    } else {
        iter_limit = ITRS_EST;
    }

    /* Collect and print the timing results recorded by each process */
    Print_Timings(time, title, p->size, iter_limit, p->comm);

    return time;
}

int main (int argc, char *argv[])
{
    int err;
    double time, time_limit, time_maxMsg;

    int iter, iter_limit;
    size_t size, messStart, messStop, mem_limit;
    int testFlags, ndims, partsize;
    int k;

    char  hostname[256];
    char* hostnames;

    int root = 0;

    struct argList args;
    /* process the command-line arguments, printing usage info on error */
    if (!processArgs(argc, argv, &args)) { usage(); }
    iter        = args.iters;
    messStart   = args.messStart;
    messStop    = args.messStop;
    mem_limit   = args.memLimit;
    time_limit  = args.timeLimit;
    testFlags   = args.testFlags;
    check_once  = args.checkOnce;
    check_every = args.checkEvery;
    ndims       = args.ndims;
    partsize    = args.partSize; 

    /* initialize MPI */
    err = MPI_Init(&argc, &argv);
    if (err) { printf("Error in MPI_Init\n"); exit(1); }

    /* determine who we are in the MPI world */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_local);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    /* mark start of mpiBench output */
    if (rank_local == 0) { printf("START mpiBench v%s\n", VERS); }

    /* collect hostnames of all the processes and print rank layout */
    gethostname(hostname, sizeof(hostname));
    hostnames = (char*) _ALLOC_MAIN_(sizeof(hostname)*rank_count, "Hostname array");
    MPI_Gather(hostname, sizeof(hostname), MPI_CHAR, hostnames, sizeof(hostname), MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank_local == 0) {
        for(k=0; k<rank_count; k++) {
            printf("%d : %s\n", k, &hostnames[k*sizeof(hostname)]);
        }
    }

    /* allocate message buffers and initailize timing functions */
    while(messStop*((size_t)rank_count)*2 > mem_limit && messStop > 0) messStop /= 2;
    buffer_size = messStop * rank_count;
    sbuffer   = (char*) _ALLOC_MAIN_(messStop    * rank_count, "Send Buffer");
    rbuffer   = (char*) _ALLOC_MAIN_(messStop    * rank_count, "Receive Buffer");
    sendcounts = (int*) _ALLOC_MAIN_(sizeof(int) * rank_count, "Send Counts");
    sdispls    = (int*) _ALLOC_MAIN_(sizeof(int) * rank_count, "Send Displacements");
    recvcounts = (int*) _ALLOC_MAIN_(sizeof(int) * rank_count, "Recv Counts");
    rdispls    = (int*) _ALLOC_MAIN_(sizeof(int) * rank_count, "Recv Displacements");

    /*time_maxMsg = 2*time_limit; */
    time_maxMsg = 0.0;

    /* if partsize was specified, calculate the number of partions we need */
    int partitions = 0;
    if (partsize > 0) {
        /* keep dividing comm in half until we get to partsize */
        int currentsize = rank_count;
        while (currentsize >= partsize) {
            partitions++;
            currentsize >>= 1;
        }
    }

    /* set up communicators */
    int total_comms = 1+ndims+partitions;
    int current_comm = 0;
    int extra_state;
    MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &dimid_key, (void*) &extra_state);
    MPI_Comm* comms = (MPI_Comm*) _ALLOC_MAIN_(sizeof(MPI_Comm) * total_comms, "Communicator array");
    char* comm_desc = (char*)     _ALLOC_MAIN_(256              * total_comms, "Communicator description array");

    /* the first communicator is MPI_COMM_WORLD */
    comms[0]  = MPI_COMM_WORLD;
    strcpy(&comm_desc[256*current_comm], "MPI_COMM_WORLD");
    MPI_Comm_set_attr(comms[0], dimid_key, (void*) &comm_desc[256*current_comm]); 
    current_comm++;

    /* if ndims is specified, map MPI_COMM_WORLD into ndims Cartesian space, and create 1-D communicators along each dimension */
    int d;
    if (ndims > 0) {
        MPI_Comm comm_dims;
        int* dims    = (int*) _ALLOC_MAIN_(sizeof(int) * ndims, "Dimension array");
        int* periods = (int*) _ALLOC_MAIN_(sizeof(int) * ndims, "Period array");
        for (d=0; d < ndims; d++) {
            dims[d]    = 0; /* set dims[d]=non-zero if you want to explicitly specify the number of processes in this dimension */
            periods[d] = 0; /* set period[d]=1 if you want this dimension to be periodic */ 
        }

        /*
        given the total number of processes, and the number of dimensions,
        fill in dims with the number of processes along each dimension (split as evenly as possible)
        */ 
        MPI_Dims_create(rank_count, ndims, dims);

        /*
        then create a cartesian communicator,
        which we'll use to split into ndims 1-D communicators along each dimension
        */  
        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &comm_dims);

        /*
        split cartesian communicator into ndims 1-D subcommunicators
        (for MPI_Cart_sub, set dims[d]=1 if you want that dimension to remain in subcommunicator)
        */ 
        int d2;
        for (d=0; d < ndims; d++) {
            for (d2=0; d2 < ndims; d2++) {
                if (d == d2) dims[d2] = 1;
                else         dims[d2] = 0;
            }
            MPI_Cart_sub(comm_dims, dims, &comms[current_comm]);
            sprintf(&comm_desc[256*current_comm], "CartDim-%dof%d", d+1, ndims);
            MPI_Comm_set_attr(comms[current_comm], dimid_key, (void*) &comm_desc[256*current_comm]); 
            current_comm++;
        }

        if (dims)    { free(dims);     dims    = NULL; }
        if (periods) { free(periods);  periods = NULL; }
    }

    /* if a partsize is specified, recursively divide MPI_COMM_WORLD in half until groups reach partsize */
    int currentsize = rank_count;
    int p = 0;
    while (p < partitions) {
        int partnum = (int) rank_local / currentsize;
        MPI_Comm_split(MPI_COMM_WORLD, partnum, rank_local, &comms[current_comm]);
        sprintf(&comm_desc[256*current_comm], "PartSize-%d", currentsize);
        MPI_Comm_set_attr(comms[current_comm], dimid_key, (void*) &comm_desc[256*current_comm]); 
        current_comm++;
        currentsize >>= 1;
        p++;
    }

    /* for each communicator, run collective tests */
    for (d=0; d < total_comms; d++) {
        MPI_Comm comm = comms[d];

        /* determine who we are in this communicator */
        int myrank, nranks;
        MPI_Comm_rank(comm, &myrank);
        MPI_Comm_size(comm, &nranks);

        struct collParams p;
        p.root   = 0;
        p.comm   = comm;
        p.myrank = myrank;
        p.nranks = nranks;
        p.type   = MPI_BYTE;

        /* time requested collectives */
        if(testFlags & BARRIER) {
            p.size = 0;
            p.count = 0;
            get_time(time_barrier, "Barrier", &p, iter, time_limit);
        }

        if(testFlags & BCAST) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                p.count = p.size;
                if(get_time(time_bcast, "Bcast", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        if(testFlags & ALLTOALL) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                p.count = p.size;
                if(get_time(time_alltoall, "Alltoall", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        if(testFlags & ALLTOALLV) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                p.count = p.size;
                if(get_time(time_alltoallv, "Alltoallv", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        if(testFlags & ALLGATHER) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                p.count = p.size;
                if(get_time(time_allgather, "Allgather", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        if(testFlags & ALLGATHERV) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                p.count = p.size;
                if(get_time(time_allgatherv, "Allgatherv", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        if(testFlags & GATHER) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                p.count = p.size;
                if(get_time(time_gather, "Gather", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        if(testFlags & GATHERV) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                p.count = p.size;
                if(get_time(time_gatherv, "Gatherv", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        if(testFlags & SCATTER) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                p.count = p.size;
                if(get_time(time_scatter, "Scatter", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        /* for the reductions, actually add some doubles to do something of interest */
        p.type     = MPI_DOUBLE;
        p.reduceop = MPI_SUM;

        if(testFlags & ALLREDUCE) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                if(p.size < sizeof(double)) continue;
                p.count = p.size / sizeof(double);
                if(get_time(time_allreduce, "Allreduce", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }

        if(testFlags & REDUCE) {
            for(p.size = messStart; p.size <= messStop; p.size = (p.size > 0) ? p.size << 1 : 1) {
                if(p.size < sizeof(double)) continue;
                p.count = p.size / sizeof(double);
                if(get_time(time_reduce, "Reduce", &p, iter, time_limit) > time_maxMsg && time_maxMsg > 0.0) break;
            }
        }
    } /* end loop over communicators */

    /* print memory usage */
    if (rank_local == 0) {
        printf("Message buffers (KB):\t%ld\n", allocated_memory/1024);
    }

#if 0
#ifndef _AIX
    print_mpi_resources();
#endif
#endif

    /* mark end of output */
    if (rank_local == 0) { printf("END mpiBench\n"); }

    /* free memory */
    if (hostnames)  { free(hostnames);  hostnames  = NULL; }

    if (sbuffer)    { free(sbuffer);    sbuffer    = NULL; }
    if (rbuffer)    { free(rbuffer);    rbuffer    = NULL; }
    if (sendcounts) { free(sendcounts); sendcounts = NULL; }
    if (sdispls)    { free(sdispls);    sdispls    = NULL; }
    if (recvcounts) { free(recvcounts); recvcounts = NULL; }
    if (rdispls)    { free(rdispls);    rdispls    = NULL; }

    if (comms)      { free(comms);      comms      = NULL; }
    if (comm_desc)  { free(comm_desc);  comm_desc  = NULL; }

    /* shut down */
    MPI_Finalize();

    return 0;
}
