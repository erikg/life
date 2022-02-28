#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "life.h"
#include "disp.h"

#define WIDTH 512
#define HEIGHT 256

void seed(Life *l) {
	static int zq;
	int j;
	unsigned char buf[256*256];

	srand(getpid());
	for(j=0;j<256*256;j++) {
		buf[j] = rand()&0x5;
	}
	l->load(buf, 64, 64, 16, 16);
}

int main(int argc, char **argv) {
	Life *l = new Life(WIDTH, HEIGHT);
	seed(l);
	Display *d = new Display(WIDTH, HEIGHT);
	while(d->input()) {
		l->sim();
		d->update(l->buffer(), l);
		d->swap();
	}
	delete d;
	return EXIT_SUCCESS;
}
