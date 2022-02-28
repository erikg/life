#ifndef _DISP_H_
#define _DISP_H_

#include "life.h"

class Display {
    public:
        Display(int width, int height);
        ~Display();
        void update(void *buf, Life *l);
        void swap();
        int input();
    private:
        int width, height;
        void *blob;
};

#endif
