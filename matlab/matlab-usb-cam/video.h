#ifndef VIDEO_H_
#define VIDEO_H_
#include <highgui.h>
#include "abstractcapturesource.h"
class Video: public AbstractCaptureSource {
public:
    int frames;
    const char* name;
    Video(const char* filename);
    Video(const char* filename, int frames);
    virtual ~Video();
    void rewind();
};
#endif /*VIDEO_H_*/
