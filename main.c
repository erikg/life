#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "life.h"
#include "disp.h"

#define WIDTH 512
#define HEIGHT 256

void seed() {
	static int zq;
	int j;
	unsigned char buf[256*256];

	srand(getpid());
	for(j=0;j<256*256;j++) {
		buf[j] = rand()&0x5;
	}
	life_load(buf, 64, 32, 32, 32);
}

int main(int argc, char **argv) {
	life_init(WIDTH,HEIGHT);
	disp_init(WIDTH, HEIGHT);
	seed();
	while(disp_input()) {
		for(int i=0;i<10;i++)
			life_sim();
		disp_update(life_buffer(), WIDTH, HEIGHT);
		disp_swap();
	}
	disp_deinit();
	life_deinit();
	return EXIT_SUCCESS;
}
