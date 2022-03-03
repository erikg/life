#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" {

#define HEIGHT 512

#include "cell.h"
#include "life.h"

#ifdef CUDA
__global__
void life_sim_row(cell_t *src, cell_t *dst, int width) {
	int y = blockIdx.x;
#else
void life_sim_row(cell_t *src, cell_t *dst, int width, int y) {
#endif
	int x;
	for(x=1;x<width;x++) {
#define val_at(_x, _y) (src[(1+_x)+(width)*(_y+1)].val)
		unsigned char live = 
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
		} else {
			dst[1+x+(width)*(y+1)].val = 0;
		}
#undef val_at
	}
	dst[0].val = dst[width+1].val = 0;
}

cell_t *tile[2];	// device
cell_t *local_tile;	// host
unsigned int currBuffer = 0;	// device
int width, height;	// device


void life_init(int width_, int height_) {
	int bytes = (width_ + 2) * (height_ + 2) * sizeof(cell_t);
	width = width_;
	height = height_;
#ifdef CUDA
	cudaMalloc(tile, bytes);
	cudaMalloc(tile+1, bytes);
#else
	tile[0] = (cell_t *)malloc(bytes);
	tile[1] = (cell_t *)malloc(bytes);
#endif
	local_tile = (cell_t *)malloc(bytes);
}

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
	for(j=0;j<h;j++) {
#ifdef CUDA
		cudaMemcpy(tile[currBuffer] + ((off_y+j+1)*width) + off_x + 1, buf+(j*w), w, cudaMemcpyHostToDevice);
#else
		// cudaMemcpy(tile[currBuffer] + ((off_y+j)*width) + off_x, buf+(j*w), w, cudaMemcpyHostToHost);
		memcpy(tile[currBuffer] + ((off_y+j+1)*width) + off_x + 1, buf+(j*w), w);
#endif
	}
}

void life_sim() {
	memset(tile[1-currBuffer], 0, (width+1)*(height+1));
#ifdef CUDA
	life_sim_row<<<HEIGHT,1>>>(tile[currBuffer], tile[1-currBuffer], width);
#else
	int y = height;
	for(y=1;y<(height+1);y++) {
		life_sim_row(tile[currBuffer], tile[1-currBuffer], width, y);
	}
	memset(tile[currBuffer], 0, width);
	memset(tile[currBuffer] + (width + 1) * height, 0, width);
	memset(tile[currBuffer] + (width + 1) * (height-1), 0, width);
#endif
	currBuffer = 1-currBuffer;
}

cell_t *life_buffer() {
#ifdef CUDA
	cudaMemcpy(local_tile, tile[currBuffer], width*height*sizeof(cell_t), cudaMemcpyDeviceToHost);
#else
//	cudaMemcpy(local_tile, tile[currBuffer], width*height, cudaMemcpyHostToHost);
	memcpy(local_tile, tile[currBuffer], width*height*sizeof(cell_t));
#endif
	return local_tile;
}

}
