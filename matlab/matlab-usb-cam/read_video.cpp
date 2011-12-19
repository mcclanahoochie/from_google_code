#include <iostream>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include "video.h"
#include "mex.h"
#include <iostream>
void parse_arguments(int nlhs, const mxArray* prhs[], int nrhs, char** filename, int* first_frame, int* last_frame, bool* color);

void mexFunction(
    int          nlhs,
    mxArray*      plhs[],
    int          nrhs,
    const mxArray* prhs[]
) {

    char* filename;
    int first_frame, last_frame;
    bool color;
    parse_arguments(nlhs, prhs, nrhs, &filename, &first_frame, &last_frame, &color);

    Video* video = NULL;

    if (first_frame < 0) {
        std::cout << "reading all video!" << std::endl;
        video = new Video(filename);
        first_frame = 1;
        last_frame = video->frames;//frames counted by grabbing all frames in video
    } else {
        std::cout << "reading frames: " << first_frame << "-" << last_frame << std::endl;
        video = new Video(filename, last_frame - first_frame);
    }



    int n_of_dims, channels;
    mwSize* dims;
    if (!color) {
        n_of_dims = 3;
        channels = 1;
        dims = new mwSize[3];
        dims[1] = video->getWidth();
        dims[0] = video->getHeight();
        dims[2] = last_frame - first_frame + 1 ;
    } else {
        n_of_dims = 4;
        channels = 3;
        dims = new mwSize[4];
        dims[1] = video->getWidth();
        dims[0] = video->getHeight();
        dims[3] = last_frame - first_frame + 1 ;
        dims[2] = 3;
    }
    IplImage* img = 0;

    //create 3 dimensional storage and assign it to output arguments
    plhs[0] = mxCreateNumericArray(n_of_dims, dims , mxUINT8_CLASS, mxREAL);
    //get pointer to data (cast necessary since data is not double, mxGetData returns a void*)
    unsigned char* video_ptr = (unsigned char*)mxGetData(plhs[0]);


    //frame counter
    int f = 0;
    //we skip frames until f == first_frame (we want to grab from here)
    //and we stop grabbing after f == last_frame
    while ((img = video->getNextFrame()) && f < last_frame) {
        ++f;
        //check if we are in the correct range (following annotation)
        if (f >= first_frame) { // && f <= last_frame ... no need to check this after last_frame we grab the last one
            video->convertFrame(video_ptr, img, color);
            //move pointer ahead one frame
            video_ptr += video->getHeight() * video->getWidth() * channels;
        }
    }
    mxFree(filename);
    return;
}
/*conversion from matlab arguments to c string*/
void parse_arguments(int nlhs, const mxArray* prhs[], int nrhs, char** filename, int* first_frame, int* last_frame, bool* color) {
    int   buflen, status;
    (*first_frame) = -1;
    (*last_frame) = -1;

    /* Check for proper number of arguments. */
//	  std::cout << nrhs << std::endl;
    if (nrhs != 2 && nrhs != 4)
    { mexErrMsgTxt("Two or four inputs required."); }
    else if (nlhs > 1)
    { mexErrMsgTxt("Too many output arguments."); }
    if (nrhs == 4) {
        (*first_frame) = mxGetScalar(prhs[2]);
        (*last_frame) = mxGetScalar(prhs[3]);
    }

    (*color) = mxGetScalar(prhs[1]);

    /* Input must be a string. */
    if (mxIsChar(prhs[0]) != 1)
    { mexErrMsgTxt("Input must be a string."); }

    /* Input must be a row vector. */
    if (mxGetM(prhs[0]) != 1)
    { mexErrMsgTxt("Input must be a row vector."); }

    /* Get the length of the input string. */
    buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0])) + 1;

    /* Allocate memory for input and output strings. */
    (*filename) = (char*)mxCalloc(buflen, sizeof(char));

    /* Copy the string data from prhs[0] into a C string
     * input_buf. */
    status = mxGetString(prhs[0], (*filename), buflen);
    if (status != 0)
    { mexWarnMsgTxt("Not enough space. String is truncated."); }
}
