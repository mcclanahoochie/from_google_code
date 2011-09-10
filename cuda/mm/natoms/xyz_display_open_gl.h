#ifndef XYZ_DISP_GL_H_
#define XYZ_DISP_GL_H_


int showGL(int argc, char* argv[], float* xyz_in, int N);
int showGLbonds(int argc, char* argv[], float* xyz_in, int N, int* nblist_in, int nbonds);
void updateNeighbors(int nbonds, int* nblist_in, int N, float* xyz_in);
void killGL();

#endif
