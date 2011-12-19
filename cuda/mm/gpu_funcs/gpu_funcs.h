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



#ifndef _GPU_FUNCS_H_
#define _GPU_FUNCS_H_

/*
  ____ ____  _   _
 / ___|  _ \| | | |
| |  _| |_) | | | |
| |_| |  __/| |_| |
 \____|_|    \___/

*/

// exposed functions in shared library 'libcudafuncs'
#ifdef __cplusplus
extern "C"
{
    int compute_bonds_gpu(float* h_xyz, int N, float rmin, float rmax, float maxrad, int nbins, int** nblist_out, int** bins_out);
    int compute_xyz_autocorrelation_gpu(float* h_xyz, int N, float& oacx, float& oacy, float& oacz, int type);
    int compute_int_autocorrelation_gpu(int* h_i, int N, float& oaci, int type);
}
#endif

#endif

