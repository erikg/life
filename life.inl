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


void life_deinit() {
#ifdef CUDA
	cudaFree(tile[0]);
	cudaFree(tile[1]);
#else
	free(tile[0]);
	free(tile[1]);
#endif
	tile[0] = tile[1] = 0;
	width = height = 0;
}

void life_load(cell_t *buf, int w, int h, int off_x, int off_y) {
	int j;
	for(j=1;j<h;j++) {
#ifdef CUDA
		cudaMemcpy(tile[currBuffer] + ((off_y+j+1)*width) + off_x + 1, buf+(j*w), w, cudaMemcpyHostToDevice);
#else
		// cudaMemcpy(tile[currBuffer] + ((off_y+j)*width) + off_x, buf+(j*w), w, cudaMemcpyHostToHost);
		memcpy(tile[currBuffer] + ((off_y+j+1)*width) + off_x + 1, buf+(j*w), w);
#endif
	}
}

cell_t *life_buffer() {
#ifdef CUDA
	cudaMemcpy(local_tile, tile[currBuffer], width*height*sizeof(cell_t), cudaMemcpyDeviceToHost);
#else
//	cudaMemcpy(local_tile, tile[currBuffer], width*height, cudaMemcpyHostToHost);
	memcpy(local_tile, tile[currBuffer], ((width)*(height+1)+1)*sizeof(cell_t));
#endif
	return local_tile;
}
