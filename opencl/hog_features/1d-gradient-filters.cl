
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

