#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cell.h"
#include "life.h"

#ifdef CUDA
__global__
void life_sim_cell(cell_t *src, cell_t *dst, int width) {
	int y = blockIdx.y;
	int x = blockIdx.x;
#else
void life_sim_cell(cell_t *src, cell_t *dst, int width, int x, int y) {
#endif

	unsigned char live = 
#define val_at(_x, _y) (src[(1+_x)+(width)*(_y+1)].val)
		val_at(x-1, y) 
		+ val_at(x+1, y) 

		+ val_at(x-1, y-1) 
		+ val_at(x, y-1)
		+ val_at(x+1, y-1) 

		+ val_at(x-1, y+1) 
		+ val_at(x, y+1) 
		+ val_at(x+1, y+1);
	if(live == 0) {
		dst[1+x+(width)*(y+1)].val = 0;
	} else if( (live&0xf) && !(live>>4) ) {
		live &= 0xf;
		dst[1+x+(width)*(y+1)].val = (val_at(x,y)?(((live)>>1)&1):(live)==3) ? 0x1 : 0;
	} else if( !(live&0xf) && (live>>4) ) {
		live >>= 4;
		dst[1+x+(width)*(y+1)].val = (val_at(x,y)?((live>>1)&1):(live)==3) ? 0x10 : 0;
#undef val_at
	} else {
		dst[1+x+(width)*(y+1)].val = 0;
	}
}

cell_t *tile[2];	// device
cell_t *local_tile;	// host
unsigned int currBuffer = 0;	// device
int width, height;	// device
