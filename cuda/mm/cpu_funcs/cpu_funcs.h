
#ifndef _CPU_FUNCS_H_
#define _CPU_FUNCS_H_

/*
  ____ ____  _   _
 / ___|  _ \| | | |
| |   | |_) | | | |
| |___|  __/| |_| |
 \____|_|    \___/

*/

// exposed functions in shared library 'libcudafuncs'
#ifdef __cplusplus
extern "C"
{
    int compute_bonds_cpu(float* h_xyz, int N, float rmin, float rmax, float maxrad, int nbins, int** nblist_out, int** bins_out);
    int compute_xyz_autocorrelation_cpu(float* h_xyz, int N, float& oacx, float& oacy, float& oacz, int type);
    int compute_int_autocorrelation_cpu(int* h_i, int N, float& oaci, int type);
}
#endif

#endif

