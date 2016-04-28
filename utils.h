#ifndef __UTILS_H__
#define __UTILS_H__
#include <stdio.h>
#include <iostream>

//For timing
// #include <time.h>
// #include <sys/time.h>

// clock for mac
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//	   std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}


// unsigned long long t1, t2;
// unsigned long long t_cpu=0, t_gpu=0;

// unsigned long long absoluteTime()
// {
//     const unsigned int nanoFactor = 1000000000;
//     mach_timespec_t res;
//     clock_serv_t cclock;
//     host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
//     clock_get_time(cclock, &res); // Here you can change parameters for physical or CPU time.
//     mach_port_deallocate(mach_task_self(), cclock);
//     unsigned long long cur_time = res.tv_sec;
//     cur_time = cur_time*nanoFactor + res.tv_nsec;
//     return cur_time;
// }

//assumes little endian
void printBits(size_t const size, void const * const ptr)
{
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;

    for (i=size-1;i>=0;i--)
    {
        for (j=7;j>=0;j--)
        {
            byte = b[i] & (1<<j);
            byte >>= j;
            printf("%u", byte);
        }
    }
    puts("");
}

void printDevInfo(int count, 
				cudaDeviceProp prop) {
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaGetDeviceProperties(&prop, i) );
		printf( "   --- General Information for device %d ---\n", i );
	    printf( "Name:  %s\n", prop.name );
	    printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
	    printf( "Clock rate:  %d\n", prop.clockRate );
	    printf( "Device copy overlap:  " );
	    if (prop.deviceOverlap)
	        printf( "Enabled\n" );
	    else
	        printf( "Disabled\n" );
	    printf( "Kernel execution timeout :  " );
	    if (prop.kernelExecTimeoutEnabled)
	        printf( "Enabled\n" );
	    else
	        printf( "Disabled\n" );
	    printf( "   --- Memory Information for device %d ---\n", i );
	    printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
	    printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
	    printf( "Max mem pitch:  %ld\n", prop.memPitch );
	    printf( "Texture Alignment:  %ld\n", prop.textureAlignment );
	    printf( "   --- MP Information for device %d ---\n", i );
	    printf( "Multiprocessor count:  %d\n",
	                prop.multiProcessorCount );
	    printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
	    printf( "Registers per mp:  %d\n", prop.regsPerBlock );
	    printf( "Threads in warp:  %d\n", prop.warpSize );
	    printf( "Max threads per block:  %d\n",
	                prop.maxThreadsPerBlock );
	    printf( "Max thread dimensions:  (%d, %d, %d)\n",
	                prop.maxThreadsDim[0], prop.maxThreadsDim[1],
	                prop.maxThreadsDim[2] );
	    printf( "Max grid dimensions: (%d, %d, %d)\n",
	                prop.maxGridSize[0], prop.maxGridSize[1],
	                prop.maxGridSize[2] );
	        printf( "\n" );
	    return;
	}
}

#endif