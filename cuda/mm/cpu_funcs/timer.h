/*
 * timer.h
 *
 *  Created on: ?, 2009
 *      Author: chris
 */

#ifndef _TIMER_H_
#define _TIMER_H_

#include <sys/time.h>
#include <sys/resource.h>

#define ERROR_VALUE -1.0
#define FALSE 0
#define TRUE  1
#define MAX_TIMERS 10

static int timer_set[MAX_TIMERS];
static long long old_time[MAX_TIMERS];


/* Return the amount of time in useconds used by the current process since it began. */
inline long long user_time() {
    struct timeval tv;
    gettimeofday(&tv, (struct timezone*) NULL);
    return ((tv.tv_sec * 1000000) + (tv.tv_usec));   // usec
}


/* Starts timer. */
inline void start_timer(int timer) {
    timer_set[timer] = TRUE;
    old_time[timer] = user_time();
}


/* Returns elapsed time since last call to start_timer().
   Returns ERROR_VALUE if Start_Timer() has never been called. */
inline double  elapsed_time(int timer) {
    if (timer_set[timer] != TRUE) {
        return (ERROR_VALUE);
    } else {
        return (user_time() - old_time[timer]) / 1000.0  ; // msec
    }
}


#endif /*_TIMER_H_*/



