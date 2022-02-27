#include <stdio.h>
#include <stdlib.h>

#include "life.h"
#include "disp.h"

#define WIDTH 1200
#define HEIGHT 600

void seed() {
	static int zq;
	int j;
	char buf[256*256];

	srand(getpid());
	for(j=0;j<256*256;j++) {
		buf[j] = rand()&0x5;
	}
	life_load(buf, 32, 128, 48, 48);
}

int main(int argc, char **argv) {
	life_init(WIDTH,HEIGHT);
	disp_init(WIDTH, HEIGHT);
	seed();
	while(disp_input()) {
		for(int i = 0; i < 10; i++) {
			life_sim();
		}
		disp_update(life_buffer(), WIDTH, HEIGHT);
		disp_swap();
	}
	disp_deinit();
	life_deinit();
	return EXIT_SUCCESS;
}
