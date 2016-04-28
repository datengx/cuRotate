//For timing
#include <time.h>
#include <sys/time.h>

// clock for mac
#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif


unsigned long long t1, t2;
unsigned long long t_cpu=0, t_gpu=0;

unsigned long long absoluteTime()
{
    const unsigned int nanoFactor = 1000000000;
    mach_timespec_t res;
    clock_serv_t cclock;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &res); // Here you can change parameters for physical or CPU time.
    mach_port_deallocate(mach_task_self(), cclock);
    unsigned long long cur_time = res.tv_sec;
    cur_time = cur_time*nanoFactor + res.tv_nsec;
    return cur_time;
}