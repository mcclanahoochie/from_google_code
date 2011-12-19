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


#define M_PI 3.14159265

__kernel void cart2polar(__global float* xpart, __global float* ypart, __global float* angle, __global float* magnitude, int rows, int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= rows || j >= cols) { return; }
    int curId = i * cols + j;
    float x = xpart[curId];
    float y = ypart[curId];
    float a = atan2(y, x);
    angle[curId] = (a < 0) ? a + 2 * M_PI : a;
    magnitude[curId] = sqrt((x * x) + (y * y));
}

