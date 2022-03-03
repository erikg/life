#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" {

#define HEIGHT 512

#include "life.h"

#ifdef CUDA
__global__
void life_sim_row(unsigned char *src, unsigned char *dst, int width) {
	int y = blockIdx.x;
#else
void life_sim_row(unsigned char *src, unsigned char *dst, int width, int y) {
#endif
	int x;
	for(x=1;x<width;x++) {
#define val_at(_x, _y) (src[(1+_x)+(width)*(_y+1)] ? 1 : 0)
		unsigned char live = 
			val_at(x-1, y) 
			+ val_at(x+1, y) 

			+ val_at(x-1, y-1) 
			+ val_at(x, y-1)
			+ val_at(x+1, y-1) 

			+ val_at(x-1, y+1) 
			+ val_at(x, y+1) 
			+ val_at(x+1, y+1);
		unsigned char newlife = val_at(x,y)?((live>>1)&1):live==3;
#undef val_at
		dst[1+x+(width)*(y+1)] = newlife?1:0;
	}
	dst[0] = dst[width+1] = 0;
}

unsigned char *tile[2];	// device
unsigned char *local_tile;	// host
unsigned char currBuffer = 0;	// device
int width, height;	// device


void life_init(int width_, int height_) {
	int bytes = (width_ + 2) * (height_ + 2);
	width = width_;
	height = height_;
#ifdef CUDA
	cudaMalloc(tile, bytes);
	cudaMalloc(tile+1, bytes);
#else
	tile[0] = (unsigned char *)malloc(bytes);
	tile[1] = (unsigned char *)malloc(bytes);
#endif
	local_tile = (unsigned char *)malloc(width * height);
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

void life_load(unsigned char *buf, int w, int h, int off_x, int off_y) {
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

void *life_buffer() {
#ifdef CUDA
	cudaMemcpy(local_tile, tile[currBuffer], width*height, cudaMemcpyDeviceToHost);
#else
//	cudaMemcpy(local_tile, tile[currBuffer], width*height, cudaMemcpyHostToHost);
	memcpy(local_tile, tile[currBuffer], width*height);
#endif
	return local_tile;
}

}
