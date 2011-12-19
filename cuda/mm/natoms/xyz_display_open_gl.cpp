#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <climits>
#include <cstring>
#include <memory>
#include <GL/freeglut.h>
#include "xyz_display_open_gl.h"
#include <pthread.h>

#define  SHOW_FPS  1

#if SHOW_FPS
#include "frames_counter.h"
#endif

using namespace std;

//
//  Global functions.
//
int showGL(int argc, char* argv[], float* xyz_in, int natoms);
void display(void);
int showGLbonds(int argc, char* argv[], float* xyz_in, int natoms, int* nblist_in, int nbonds);
void displayBonds(void);
void myinit(void);
void myReshape(int w, int h);
float r8_max(float x, float y);
float r8_min(float x, float y);
float* r83vec_max(int n, float a[]);
float* r83vec_min(int n, float a[]);
void spin_image();
void xyz_data_print(int point_num, float xyz[]);
void xyz_data_read(string input_filename, int point_num, float xyz[]);
void xyz_header_print(int point_num);
void xyz_header_read(string input_filename, int* point_num);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void keyboard(unsigned char key, int x, int y);
void keyboard2(int key, int x, int y);
void getRanges();
void getScale();
void createBuffer(int natoms, float* xyz_in);
void cleanupGL();
void updateNeighbors(int nbonds, int* nblist_in, int natoms, float* xyz_in);
void* startGL(void*);
void* startGLbonds(void*);
void killGL(void);
void drawAxes(float len);
void makeSphere(void);
void drawBonds(void);
void drawAtoms(void);


//
//  Global data.
//
#define WIDTH   800
#define HEIGHT  800
#define DIM_NUM  3
int point_num = 0;
float* xyz = NULL;
float* xyz_max = NULL;
float* xyz_min = NULL;
float xyz_range[3];
float xyz_center[3];
float xyz_scale = -1;
int mouseOldX, mouseOldY;
int mouse_buttons = 0;
float rotateX = 0.0f, rotateY = 0.0f;
float translateX = 0.0f, translateY = 0.0f, translateZ = -2.0f;
int* nblist = NULL;
int nb_num = 0;
static pthread_t gl_thread;
int argc;
char** argv;
volatile int canDraw = 0;
volatile int isDrawing = 0;
float sphere_r;
#define NUM_DISPLAY_MODES 2
int _display = 0;
enum {SPHERE = 1};


////////////////////////////////////////////////////////////////////////////////
//! Core functions
////////////////////////////////////////////////////////////////////////////////

//****************************************************************************80
int showGLbonds(int argc0, char* argv0[], float* xyz_in, int natoms, int* nblist_in, int nbonds) {
    //
    //  Initialize
    //
    updateNeighbors(nbonds, nblist_in, natoms, xyz_in);
    argc = argc0;
    argv = argv0;

    //
    //  GO!
    //
    pthread_create(&gl_thread, NULL, startGLbonds, NULL); // <---
    return 0;
}

//****************************************************************************80
int showGL(int argc0, char* argv0[], float* xyz_in, int natoms) {
    //
    //  Initialize
    //
    createBuffer(natoms, xyz_in);
    argc = argc0;
    argv = argv0;

    //
    //  GO!
    //
    pthread_create(&gl_thread, NULL, startGL, NULL); // <---
    return 0;
}

//****************************************************************************80
void* startGLbonds(void*) {
    //
    //  Initialize GLUT
    //
    {
        //
        //  GLUT calls must go here
        //
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(WIDTH, HEIGHT);
        glutInitWindowPosition(10, 10);
        glutCreateWindow("n atoms");
        glutReshapeFunc(myReshape);
        glutDisplayFunc(displayBonds); // <---
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutKeyboardFunc(keyboard);
        glutSpecialFunc(keyboard2);
        atexit(killGL);

        //
        //  GL calls can go here
        //
        myinit();
    }

    //
    //  GO!
    //
    canDraw = 1;
    glutMainLoop();

    //
    //  Never get here
    //
    pthread_exit(NULL);
}

//****************************************************************************80
void* startGL(void*) {
    //
    //  Initialize GLUT
    //
    {
        //
        //  GLUT calls must go here
        //
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(WIDTH, HEIGHT);
        glutInitWindowPosition(10, 10);
        glutCreateWindow("n atoms");
        glutReshapeFunc(myReshape);
        glutDisplayFunc(display); // <---
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutKeyboardFunc(keyboard);
        glutSpecialFunc(keyboard2);
        atexit(killGL);

        //
        //  GL calls can go here
        //
        myinit();
    }

    //
    //  GO!
    //
    canDraw = 1;
    glutMainLoop();

    //
    //  Never get here
    //
    pthread_exit(NULL);
}


////////////////////////////////////////////////////////////////////////////////
//! External hooks
////////////////////////////////////////////////////////////////////////////////

// (for the school computers)
//****************************************************************************80
extern "C"
{
    int showGL_extern(int argc, char* argv[], float* xyz_in, int natoms) {
        return showGL(argc, argv, xyz_in, natoms);
    }

    int showGLbonds_extern(int argc, char* argv[], float* xyz_in, int natoms, int* nblist_in, int nbonds) {
        return showGLbonds(argc, argv, xyz_in, natoms, nblist_in, nbonds);
    }

    void updateNeighbors_extern(int nbonds, int* nblist_in, int natoms, float* xyz_in) {
        updateNeighbors(nbonds, nblist_in, natoms, xyz_in);
    }

    void killGL_extern() {
        killGL();
    }

}

////////////////////////////////////////////////////////////////////////////////
//! Display functions
////////////////////////////////////////////////////////////////////////////////

//****************************************************************************80
void displayBonds(void) {
    //
    //  Don't access the buffer when it's being updated
    //
    if (canDraw) {
        //
        //  Disable updates
        //
        isDrawing = 1;

        //
        //  Clear the window.
        //
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //
        //  Calculate FPS
        //
#if SHOW_FPS
        frameStart();
#endif

        //
        //  draw X Y X axis in bottom left corner
        //
        drawAxes(0.5f);

        //
        //  create new matrix
        //
        glPushMatrix();

        //
        //  Set view matrix
        //
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(translateX, translateY, translateZ);
        gluLookAt(
            0.0f, 0.0f, 1.0f, /* position of eye */
            0.0f, 0.0f, 0.0f, /* at, where pointing at */
            0.0f, 1.0f, 0.0f  /* up vector of the camera */
        );
        glRotatef(rotateX, 1.0f, 0.0f, 0.0f);
        glRotatef(rotateY, 0.0f, 1.0f, 0.0f);

        //
        //  Draw the bonds in BLUE
        //
        if (_display != 1) {
            drawBonds();
        }

        //
        //  Draw the atoms in RED.
        //
        if (_display != 2) {
            drawAtoms();
        }

        //
        //  restore matrix
        //
        glPopMatrix();

        //
        //  Clear all the buffers.
        //
        glFlush();

        //
        //  Calculate FPS
        //
#if SHOW_FPS
        frameEnd(GLUT_BITMAP_HELVETICA_10, 0.35f, 0.35f, 0.35f, 0.04f, 0.94f);
#endif

        //
        //  Switch between the two buffers for fast animation.
        //
        glutSwapBuffers();

        //
        //  Enable updates
        //
        isDrawing = 0;
    }

    //
    //  Refresh screen
    //
    glutPostRedisplay();


    return;
}

//****************************************************************************80
void display(void) {
    //
    //  Don't access the buffer when it's being updated
    //
    if (canDraw) {
        //
        //  Disable updates
        //
        isDrawing = 1;

        //
        //  Clear the window.
        //
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //
        //  Calculate FPS
        //
#if SHOW_FPS
        frameStart();
#endif

        //
        //  draw X Y X axis in bottom left corner
        //
        drawAxes(0.5f);

        //
        //  create new matrix
        //
        glPushMatrix();

        //
        //  Set view matrix
        //
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(translateX, translateY, translateZ);
        gluLookAt(
            0.0f, 0.0f, 1.0f, /* position of eye */
            0.0f, 0.0f, 0.0f, /* at, where pointing at */
            0.0f, 1.0f, 0.0f  /* up vector of the camera */
        );
        glRotatef(rotateX, 1.0f, 0.0f, 0.0f);
        glRotatef(rotateY, 0.0f, 1.0f, 0.0f);

        //
        //  Draw the atoms in RED.
        //
        drawAtoms();

        //
        //  restore matrix
        //
        glPopMatrix();

        //
        //  Clear all the buffers.
        //
        glFlush();

        //
        //  Calculate FPS
        //
#if SHOW_FPS
        frameEnd(GLUT_BITMAP_HELVETICA_10, 0.5f, 0.5f, 0.5f, 0.05f, 0.95f);
#endif

        //
        //  Switch between the two buffers for fast animation.
        //
        glutSwapBuffers();

        //
        //  Enable updates
        //
        isDrawing = 0;
    }

    //
    //  Refresh screen
    //
    glutPostRedisplay();

    return;
}

////////////////////////////////////////////////////////////////////////////////
//! Drawing functions
////////////////////////////////////////////////////////////////////////////////

//****************************************************************************80
void drawBonds() {
    //
    //  Get bounds
    //
    int N = point_num * DIM_NUM - 2;
    int A = nb_num - 1;
    //
    //  Draw
    //
    glColor3f(0.1f, 0.1f, 0.7f);
    int idx1, idx2;
    float x1, y1, z1, x2, y2, z2;
    float* data = xyz;
    glBegin(GL_LINES);
    for (int atom = 0; atom < A; atom += 2) {
        idx1 = nblist[atom  ] * DIM_NUM;
        idx2 = nblist[atom + 1] * DIM_NUM;
        if (idx1 < N && idx1 >= 0 && idx2 < N && idx2 >= 0) { // ??
            x1 = *(data + idx1);   //	x1 = xyz[idx1  ];
            y1 = *(data + idx1 + 1); //	y1 = xyz[idx1+1];
            z1 = *(data + idx1 + 2); //	z1 = xyz[idx1+2];
            x2 = *(data + idx2);   //	x2 = xyz[idx2  ];
            y2 = *(data + idx2 + 1); //	y2 = xyz[idx2+1];
            z2 = *(data + idx2 + 2); //	z2 = xyz[idx2+2];

            glVertex3f(x1, y1, z1); // origin of the line
            glVertex3f(x2, y2, z2); // ending point of the line
        }
    }
    glEnd();
}

//****************************************************************************80
void drawAtoms() {
    //
    //  Get bounds
    //
    int N = point_num * DIM_NUM - 2;
    //
    //  Draw
    //
    glColor3f(0.9f, 0.1f, 0.1f);
    float p[3];
    //float r = sphere_r;
    for (int atom = 0; atom < N; atom += 3) {
        p[0] = xyz[  atom];
        p[1] = xyz[1 + atom];
        p[2] = xyz[2 + atom];
        glPushMatrix();
        glTranslatef(p[0], p[1], p[2]);
        //glutSolidSphere(r,5,5);
        glCallList(SPHERE); // Draw Sphere
        glPopMatrix();
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////

//****************************************************************************80
void mouse(int button, int state, int x, int y) {
    // which button
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1 << button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }
    // update
    mouseOldX = x;
    mouseOldY = y;
    glutPostRedisplay();
}

//****************************************************************************80
void motion(int x, int y) {
    float dx, dy;
    dx = x - mouseOldX;
    dy = y - mouseOldY;
    // which button
    if (mouse_buttons & 1) {
        // left click
        rotateX += dy * 0.15f;
        rotateY += dx * 0.15f;
    } else if (mouse_buttons & 4) {
        // right click
        translateZ += dy * 0.005f;
    } else if (mouse_buttons & 2) {
        // middle click
        translateX += dx * 0.005f;
        translateY -= dy * 0.005f;
    }
    // update
    mouseOldX = x;
    mouseOldY = y;
    glutPostRedisplay();
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard event handlers
////////////////////////////////////////////////////////////////////////////////

//****************************************************************************80
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {
    case (27):
        exit(0);
        break;
    case ('d'):
        ++_display;
        if (_display > NUM_DISPLAY_MODES) { _display = 0; }
    default:
        break;
    }
}

void keyboard2(int key, int /*x*/, int /*y*/) {
    switch (key) {
    case (GLUT_KEY_LEFT):
        break;
    case (GLUT_KEY_RIGHT):
        break;
    case (GLUT_KEY_UP):
        sphere_r += 0.001;
        if (sphere_r > 0.05f) { sphere_r = 0.05f; }
        makeSphere();
        break;
    case (GLUT_KEY_DOWN):
        sphere_r -= 0.001;
        if (sphere_r < 0.001f) { sphere_r = 0.001f; }
        makeSphere();
        break;
    default:
        break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Utility functions
////////////////////////////////////////////////////////////////////////////////

void initGL() {
    /*  */
    glClearColor(0.98, 0.98, 0.98, 1.0);				// Set background
    glShadeModel(GL_SMOOTH);							// Smooth transitions between edges
    /* light properties */
    GLfloat light_position1[] = {1.0f,  1.0f,  1.0f, 0.0f};			// Define light source position
    GLfloat ambient[] = {0.5f, 0.5f, 0.5f, 1.0f};			// Define ambient lightning
    GLfloat whiteDiffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};	// Define diffuse lighting
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  whiteDiffuse);		// Set light1 properties
    glLightfv(GL_LIGHT0, GL_POSITION, light_position1);		// Set light1 properties
    /* enable */
    glEnable(GL_COLOR_MATERIAL);// Enable color
    glEnable(GL_LIGHTING);		// Enable lighting for surfaces
    glEnable(GL_LIGHT0);		// Enable light source
    glEnable(GL_AUTO_NORMAL);	// Generates the normals to the surfaces
    //glEnable(GL_NORMALIZE);		// Keep lighting stable when scaling (can be slower)
    //glEnable(GL_CULL_FACE);		// Enabling backface culling
    glEnable(GL_DEPTH_TEST);	// check the Z-buffer before placing pixels onto the screen.
    /* misc */
    glDepthMask(GL_TRUE);		// place depth values into the Z-buffer.
    glDepthFunc(GL_LEQUAL);		// valid depth check
    glClearDepth(1.0f);			// 0 is near, 1 is far
    glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST);		// Fastest rendering chosen
    glHint(GL_POINT_SMOOTH_HINT, GL_FASTEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_FASTEST);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
    /* size of bonds and spheres */
    glLineWidth(3);
    glPointSize(2);
    sphere_r = 0.009;
}

//****************************************************************************80
void myinit(void) {
    initGL();
    makeSphere();
    return;
}

//****************************************************************************80
void makeSphere() {
    GLUquadricObj* sphere_o;
    GLfloat sphere_mat[] = {0.85f, 0.85f, 0.85f, 1.0f};
    sphere_o = gluNewQuadric();
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, sphere_mat);
    // make display lists for sphere for efficiency
    glNewList(SPHERE, GL_COMPILE);
    gluSphere(sphere_o, sphere_r, 6, 6);
    glEndList();
    gluDeleteQuadric(sphere_o);
}

//****************************************************************************80
void killGL() {
    pthread_join(gl_thread, NULL);
    cleanupGL();
}

//****************************************************************************80
void updateNeighbors(int nbonds, int* nblist_in, int natoms, float* xyz_in) {
    // disable drawing
    canDraw = 0;
    // wait untill done drawing
    while (isDrawing) { usleep(20); }
    // update xyz buffer
    createBuffer(natoms, xyz_in);
    // update neighbor buffer
    nb_num = nbonds << 1;
    if (nblist) {
        delete [] nblist;
        nblist = NULL;
    }
    nblist = new int[nb_num];
    memcpy(nblist, nblist_in, nb_num * sizeof(int));
    // enable drawing
    canDraw = 1;
}

//****************************************************************************80
void scaleData() {
    if (xyz_scale < 0) {
        getRanges();
        getScale();
    }

    int N = point_num * DIM_NUM;
    for (int i = 0; i < N; ++i) {
        xyz[i] *= xyz_scale;
    }

}

//****************************************************************************80
void createBuffer(int natoms, float* xyz_in) {
    point_num = natoms; // number of atoms
    int N = natoms * DIM_NUM;
    //  Copy incoming data into buffer
    if (xyz) {
        delete [] xyz;
        xyz = NULL;
    }
    xyz = new float[N];
    memcpy(xyz, xyz_in, N * sizeof(float));
    // scale xyz data for display
    scaleData();
}

//****************************************************************************80
void getRanges() {
    //
    //  Get ranges
    //
    xyz_min = r83vec_min(point_num, xyz);
    xyz_max = r83vec_max(point_num, xyz);
    xyz_range[0] = xyz_max[0] - xyz_min[0];
    xyz_range[1] = xyz_max[1] - xyz_min[1];
    xyz_range[2] = xyz_max[2] - xyz_min[2];
    //
    //  Display
    //
    //xyz_data_print ( point_num, xyz );
    printf("\nThe number of points = %d \n\n", point_num);
    printf("              X     Y     Z \n");
    printf("  Minimum: %0.3f %0.3f %0.3f \n", xyz_min[0],   xyz_min[1],   xyz_min[2]);
    printf("  Maximum: %0.3f %0.3f %0.3f \n", xyz_max[0],   xyz_max[1],   xyz_max[2]);
    printf("  Range:   %0.3f %0.3f %0.3f \n", xyz_range[0], xyz_range[1], xyz_range[2]);
    //
    //  Sanity checks
    //
    if (xyz_range[0] == 0.0f) {
        cout << "\n";
        cout << "XYZL_DISPLAY_OPEN_GL - Fatal error!\n";
        cout << "  The X data range is 0.\n";
        //exit ( 1 );
    }
    if (xyz_range[1] == 0.0f) {
        cout << "\n";
        cout << "XYZL_DISPLAY_OPEN_GL - Fatal error!\n";
        cout << "  The Y data range is 0.\n";
        //exit ( 1 );
    }
    if (xyz_range[2] == 0.0f) {
        cout << "\n";
        cout << "XYZL_DISPLAY_OPEN_GL - Fatal error!\n";
        cout << "  The Z data range is 0.\n";
        //exit ( 1 );
    }
    cout << "\n";
}

//****************************************************************************80
void getScale() {
    //
    //  Scale the data so it fits in the unit cube.
    //
    xyz_scale = 0.0f;
    for (int dim = 0; dim < DIM_NUM; ++dim) {
        xyz_center[dim] = (xyz_min[dim] + xyz_max[dim]) / 2.0f;
        xyz_scale = r8_max(xyz_scale, (xyz_max[dim] - xyz_min[dim]) / 2.0f);
    }
    xyz_scale = sqrt(2.0f) * xyz_scale;
    printf("  Scale:   %0.3f \n\n", xyz_scale);

    xyz_scale = 1.0f / xyz_scale; // invert to save future divisions
}

//****************************************************************************80
void drawAxes(float len) {
    glDisable(GL_LIGHTING);
    glPushMatrix();

    //
    //  Set view matrix
    //
    glTranslatef(-1.4f, -1.4f, 0.0f);
    gluLookAt(
        0.0f, 0.0f, 1.0f, /* position of eye */
        0.0f, 0.0f, 0.0f, /* at, where pointing at */
        0.0f, 1.0f, 0.0f  /* up vector of the camera */
    );
    glRotatef(rotateX, 1.0f, 0.0f, 0.0f);
    glRotatef(rotateY, 0.0f, 1.0f, 0.0f);

    //
    //  draw x y z axis
    //
    glBegin(GL_LINES);
    // x - red
    glColor3f(1, 0, 0);
    glVertex3d(0.0f, 0.0f, 0.0f);
    glVertex3d(len, 0.0f, 0.0f);
    // y - green
    glColor3f(0, 1, 0);
    glVertex3d(0.0f, 0.0f, 0.0f);
    glVertex3d(0.0f, len, 0.0f);
    // x - blue
    glColor3f(0, 0, 1);
    glVertex3d(0.0f, 0.0f, 0.0f);
    glVertex3d(0.0f, 0.0f, len);
    glEnd();

    //
    //  drax axis labels
    //
    len += 0.01f;
    glColor3f(0.5f, 0.5f, 0.5f);
    glRasterPos3d(len, 0.0f, 0.0f);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'X');
    glRasterPos3d(0.0f, len, 0.0f);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'Y');
    glRasterPos3d(0.0f, 0.0f, len);
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, 'Z');


    glPopMatrix();
    glEnable(GL_LIGHTING);
}

//****************************************************************************80
void cleanupGL() {
    //
    //  Things that won't actually happen because we never return from glutMainLoop:
    //
    if (xyz) { delete [] xyz; }
    if (nblist) { delete [] nblist; }
    cout << "\n";
    cout << "XYZ_DISPLAY_OPEN_GL:\n";
    cout << "  Normal end of execution.\n";
    cout << "\n";
}

//****************************************************************************80
void myReshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //
    //  set perspective
    //
    gluPerspective(
        55.0f,		// deg FOV
        (float)w / (float)h,	// aspect ratio
        0.01f, 		// z near
        5.0f		// z far
    );
}

//****************************************************************************80
// left over from modified example
//****************************************************************************80
float r8_max(float x, float y) {
    float value;
    if (y < x) { value = x; }
    else { value = y; }
    return value;
}

//****************************************************************************80
float r8_min(float x, float y) {
    float value;
    if (y < x) { value = y; }
    else { value = x; }
    return value;
}

//****************************************************************************80
float* r83vec_max(int n, float a[]) {
    float* amax = NULL;
    int i;
    int j;
    if (n <= 0) { return NULL; }
    amax = new float[DIM_NUM];
    for (i = 0; i < DIM_NUM; ++i) {
        //amax[i] = a[i+0*DIM_NUM];
        amax[i] = a[i];
        for (j = 1; j < n; ++j) {
            if (amax[i] < a[i + j * DIM_NUM]) {
                amax[i] = a[i + j * DIM_NUM];
            }
        }
    }
    return amax;
}

//****************************************************************************80
float* r83vec_min(int n, float a[]) {
    float* amin = NULL;
    int i;
    int j;
    if (n <= 0) { return NULL; }
    amin = new float[DIM_NUM];
    for (i = 0; i < DIM_NUM; ++i) {
        //amin[i] = a[i+0*DIM_NUM];
        amin[i] = a[i];
        for (j = 1; j < n; ++j) {
            if (a[i + j * DIM_NUM] < amin[i]) {
                amin[i] = a[i + j * DIM_NUM];
            }
        }
    }
    return amin;
}

//****************************************************************************80
void xyz_data_print(int point_num, float xyz[]) {
    int j;
    cout << "\n";
    for (j = 0; j < point_num; ++j) {
        cout << setw(10) << xyz[0 + j * 3] << "  "
             << setw(10) << xyz[1 + j * 3] << "  "
             << setw(10) << xyz[2 + j * 3] << "\n";

    }
    return;
}

//****************************************************************************80
void xyz_header_print(int point_num) {
    cout << "\n";
    cout << "  Number of points = " << point_num << "\n";
    return;
}


//****************************************************************************80


