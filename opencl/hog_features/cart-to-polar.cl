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

