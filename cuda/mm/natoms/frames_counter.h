/*****************************************

 frames.h

 General purpose frames per second counter for OpenGL/GLUT GNU/Linux and
 Windows programs. Displays "Frames per second: N" at an arbitrary position
 in the window. Saves and restores the app's modelview and projection
 matrices, colour, and lighting.

 Author: Toby Howard. toby@cs.man.ac.uk.
 Modified to work under MSVC by Matt Craven. cravenm7@cs.man.ac.uk

 Version 2.1, 24 March 1999

 Important MSVC issues:
 Windows' timer operates in milliseconds rather than microseconds,
 so my code compensates for this by adding an extra millisecond
 onto the end time. This is because a simple scene will render in
 less than 1 millisecond, which would cause a divide by 0.  So if
 you see an fps of 1000, it really means that it couldn't be timed,
 since your graphics are all being drawn in under 1 millisecond.

 ====================

 Usage: to add an on-screen frames per second counter to your program, save
 this file alongside your app as "frames.h", and add:

    #include "frames.h"

 immediately after all your app's other #includes; then bracket all the
 code in your display() function, before swapping buffers, with

   frameStart();

 and

   frameEnd(void *font, GLclampf r, GLclampf g, GLclampf b,
            float x, float y);

     font:    font to use, e.g., GLUT_BITMAP_HELVETICA_10
     r, g, b: RGB text colour
     x, y:    text position in window: range [0,0] (bottom left of window)
              to [1,1] (top right of window).

 ====================

 Example:

    void display(void) {
      glClear(GL_COLOR_BUFFER_BIT);

      frameStart();

      // all the graphics code

      frameEnd(GLUT_BITMAP_HELVETICA_10, 1.0, 1.0, 1.0, 0.05, 0.95);

      glutSwapBuffers();
    }
*****************************************/

#ifndef __WIN32__
#include <sys/time.h>
#define ELAPSED frameEndTime.tv_sec - frameStartTime.tv_sec + \
             ((frameEndTime.tv_usec - frameStartTime.tv_usec)/1.0E6);
#else
#include <time.h>
/* Structure taken from the BSD file sys/time.h. */
typedef struct timeval {
    long    tv_sec;         /* seconds */
    long    tv_usec;        /* and microseconds */
} timeval;

/* Replacement gettimeofday
   It really just sets the microseconds to the clock() value
   (which under Windows is really milliseconds) */
void gettimeofday(timeval* t, void* __not_used_here__) {
    t->tv_usec = (long)(clock());
}

#define ELAPSED (float)(frameEndTime.tv_usec+1 - frameStartTime.tv_usec)/1.0E3
#endif
#include <stdio.h>

struct timeval frameStartTime, frameEndTime;

void frameStart(void) {
    gettimeofday(&frameStartTime, NULL);
}

void frameEnd(void* font, GLclampf r, GLclampf g, GLclampf b,
              GLfloat x, GLfloat y) {
    /* font: font to use, e.g., GLUT_BITMAP_HELVETICA_10
       r, g, b: text colour
       x, y: text position in window: range [0,0] (bottom left of window)
             to [1,1] (top right of window). */

    float elapsedTime;
    char str[32];
    char* ch;
    GLint matrixMode;
    GLboolean lightingOn;
    gettimeofday(&frameEndTime, NULL);
    elapsedTime = ELAPSED;
    sprintf(str, "Frames per second: %2.0f", 1.0f / elapsedTime);
    lightingOn = glIsEnabled(GL_LIGHTING);       /* lighting on? */
    if (lightingOn) { glDisable(GL_LIGHTING); }
    glGetIntegerv(GL_MATRIX_MODE, &matrixMode);  /* matrix mode? */
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0f, 1.0f, 0.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glPushAttrib(GL_COLOR_BUFFER_BIT);       /* save current colour */
    glColor3f(r, g, b);
    glRasterPos3f(x, y, 0.0f);
    for (ch = str; *ch; ++ch) {
        glutBitmapCharacter(font, (int)*ch);
    }
    glPopAttrib();
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(matrixMode);
    if (lightingOn) { glEnable(GL_LIGHTING); }
}

/* end of frames.h */

