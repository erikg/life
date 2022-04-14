#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cell.h"
#include "life.h"
#include "disp.h"

#define SEED_EDGE 16
#define SEED_EDGE2 (SEED_EDGE*SEED_EDGE)

void seed() {
	static int zq;
	int j;
	cell_t buf[SEED_EDGE2];

	srand(getpid());
	for(j=0;j<SEED_EDGE2;j++) {
		buf[j].val = (rand()&0x5) ? 0x01 : 0;
	}
	life_load(buf, SEED_EDGE, SEED_EDGE, 2, 2);

	for(j=0;j<SEED_EDGE2;j++) {
		buf[j].val = (rand()&0x5) ? 0x10 : 0;
	}
	life_load(buf, SEED_EDGE, SEED_EDGE, WIDTH-SEED_EDGE-2, HEIGHT-SEED_EDGE-2);
}

int main(int argc, char **argv) {
	life_init(WIDTH,HEIGHT);
	disp_init(WIDTH, HEIGHT);
	seed();
	disp_update(life_buffer(), WIDTH, HEIGHT);
	disp_swap();
	while(disp_input()) {
		int i;
		for(i=0;i<10;i++)
			life_sim();
		disp_update(life_buffer(), WIDTH, HEIGHT);
		disp_swap();
	}
	disp_deinit();
	life_deinit();
	return EXIT_SUCCESS;
}
