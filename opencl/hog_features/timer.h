/*
   Copyright [2011] [Chris McClanahan]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


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



