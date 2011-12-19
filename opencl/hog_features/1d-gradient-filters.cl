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



__kernel void xfilter(__global float* img, __global float* xgrad, int rows, int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= rows || j >= cols) { return; }
    int curId = i * cols + j;
    int right = curId + 1;
    int left = curId - 1;
    if (j >= cols - 1) {
        right = curId;
    } else if (j < 1) {
        left = curId;
    }
    xgrad[curId] = (img[right] - img[left]);
}

__kernel void yfilter(__global float* img, __global float* ygrad, int rows, int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= rows || j >= cols) { return; }
    int curId = i * cols + j;
    int up = curId - cols;
    int down = curId + cols;
    if (i >= rows - 1) {
        down = curId;
    } else if (i < 1) {
        up = curId;
    }
    ygrad[curId] = (img[down] - img[up]);
}

