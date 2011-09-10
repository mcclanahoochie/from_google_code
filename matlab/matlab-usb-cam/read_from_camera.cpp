#include <iostream>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include "camera.h"
#include "mex.h"
#include <iostream>
void parse_arguments(int nlhs, const mxArray* prhs[], int nrhs, int* frames, bool* color);


Camera camera(0);


void mexFunction(
    int          nlhs,
    mxArray*      plhs[],
    int          nrhs,
    const mxArray* prhs[]
) {
    int frames;
    bool color;

    parse_arguments(nlhs, prhs, nrhs, &frames, &color);

//  Camera camera(0);

    if (frames < 0) {
        frames = 1;
    }

    int n_of_dims, channels;
    mwSize* dims;

    if (!color) {
        n_of_dims = 3;
        channels = 1;
        dims = new mwSize[3];
        dims[1] = camera.getWidth();
        dims[0] = camera.getHeight();
        dims[2] = frames ;
    } else {
        n_of_dims = 4;
        channels = 3;
        dims = new mwSize[4];
        dims[1] = camera.getWidth();
        dims[0] = camera.getHeight();
        dims[3] = frames;
        dims[2] = 3;
    }

    IplImage* img = 0;

    //create 3 or 4 dimensional storage and assign it to output arguments
    plhs[0] = mxCreateNumericArray(n_of_dims, dims , mxUINT8_CLASS, mxREAL);

    //get pointer to data (cast necessary since data is not double, mxGetData returns a void*)
    unsigned char* video_ptr = (unsigned char*)mxGetData(plhs[0]);

    if (frames == 3) {

        printf("sat %f \n", cvGetCaptureProperty(camera.capture, CV_CAP_PROP_SATURATION));
        printf("con %f \n", cvGetCaptureProperty(camera.capture, CV_CAP_PROP_CONTRAST));
        //cvSetCaptureProperty(camera.capture, CV_CAP_PROP_SATURATION, 0.23);
        //cvSetCaptureProperty(camera.capture, CV_CAP_PROP_CONTRAST,   0.08);



        cvSetCaptureProperty(camera.capture, CV_CAP_PROP_BRIGHTNESS, 0.05);
        camera.getNextFrame();
        camera.getNextFrame();
        camera.getNextFrame();
        camera.getNextFrame();
        img = camera.getNextFrame();
        camera.convertFrame(video_ptr, img, color);
        video_ptr += camera.getHeight() * camera.getWidth() * channels;


        cvSetCaptureProperty(camera.capture, CV_CAP_PROP_BRIGHTNESS, 0.45);
        camera.getNextFrame();
        camera.getNextFrame();
        camera.getNextFrame();
        camera.getNextFrame();
        img = camera.getNextFrame();
        camera.convertFrame(video_ptr, img, color);
        video_ptr += camera.getHeight() * camera.getWidth() * channels;


        cvSetCaptureProperty(camera.capture, CV_CAP_PROP_BRIGHTNESS, 0.99);
        camera.getNextFrame();
        camera.getNextFrame();
        camera.getNextFrame();
        camera.getNextFrame();
        img = camera.getNextFrame();
        camera.convertFrame(video_ptr, img, color);
        video_ptr += camera.getHeight() * camera.getWidth() * channels;


        cvSetCaptureProperty(camera.capture, CV_CAP_PROP_BRIGHTNESS, 0.45);
        camera.getNextFrame();
        camera.getNextFrame();
        camera.getNextFrame();
        camera.getNextFrame();


    } else {

        //frame counter
        int f = 0;
        //video pointer
        while ((img = camera.getNextFrame()) && f < frames) {
            ++f;
            camera.convertFrame(video_ptr, img, color);
            //move pointer 1 frame ahead
            video_ptr += camera.getHeight() * camera.getWidth() * channels;
        }

    }

    return;
}


/*conversion from matlab arguments to c string*/
void parse_arguments(int nlhs, const mxArray* prhs[], int nrhs, int* frames, bool* color) {
    (*frames) = -1;

    /* Check for proper number of arguments. */
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (nrhs == 1) {
        (*frames) = mxGetScalar(prhs[0]);
        (*color) = false;
    } else if (nrhs == 2) {
        (*frames) = mxGetScalar(prhs[0]);
        (*color) =  mxGetScalar(prhs[1]);
    }
}
