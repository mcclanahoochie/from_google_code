#ifndef TIMER_H_
#define TIMER_H_

#include <sys/time.h>
#include <sys/resource.h>

#define ERROR_VALUE -1.0
#define FALSE 0
#define TRUE  1
#define MAX_TIMERS 10

static int timer_set[MAX_TIMERS];
static long long old_time[MAX_TIMERS];


/* Return the amount of time in useconds used by the current process since it began. */
long long user_time() {
    struct timeval tv;
    gettimeofday(&tv, (struct timezone*) NULL);
    return ((tv.tv_sec * 1000000) + (tv.tv_usec));   // usec
}


/* Starts timer. */
void start_timer(int timer) {
    timer_set[timer] = TRUE;
    old_time[timer] = user_time();
}


/* Returns elapsed time since last call to start_timer().
   Returns ERROR_VALUE if Start_Timer() has never been called. */
double  elapsed_time(int timer) {
    if (timer_set[timer] != TRUE) {
        return (ERROR_VALUE);
    } else {
        return (user_time() - old_time[timer]) / 1000.0  ; // msec
    }
}


#endif /*TIMER_H_*/



