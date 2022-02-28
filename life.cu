#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" {

#define HEIGHT 512

#include "life.h"

// #undef CUDA
#define CUDA 1


#ifdef CUDA
__global__
void life_sim_row(unsigned char *src, unsigned char *dst, int width) {
	int y = blockIdx.x;
#else
void life_sim_row(unsigned char *src, unsigned char *dst, int width, int y) {
#endif
	int x = width;
	while(--x) {
#define val_at(_x, _y) (((_x)<0 || (_y)<0 || (_x)>(width-1) || (_y)>(width-1)) ? 0 : src[(_x)+width*(_y)] ? 1 : 0)
		unsigned char live = val_at(x-1, y) + val_at(x+1, y) + val_at(x-1, y-1) + val_at(x, y-1) \
							 + val_at(x+1, y-1) + val_at(x-1, y+1) + val_at(x, y+1) + val_at(x+1, y+1);
		unsigned char newlife = val_at(x,y)?((live>>1)&1):live==3;
#undef val_at
		dst[x+width*y] = newlife?255:0;
	}
}

unsigned char *tile[2];	// device
unsigned char *local_tile;	// host
unsigned char currBuffer = 0;	// device
int width, height;	// device


void life_init(int width_, int height_) {
	width = width_;
	height = height_;
#ifdef CUDA
	cudaMalloc(tile, width*height);
	cudaMalloc(tile+1, width*height);
#else
	tile[0] = (unsigned char *)malloc(width * height);
	tile[1] = (unsigned char *)malloc(width * height);
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
		cudaMemcpy(tile[currBuffer] + ((off_y+j)*width) + off_x, buf+(j*w), w, cudaMemcpyHostToDevice);
#else
		// cudaMemcpy(tile[currBuffer] + ((off_y+j)*width) + off_x, buf+(j*w), w, cudaMemcpyHostToHost);
		memcpy(tile[currBuffer] + ((off_y+j)*width) + off_x, buf+(j*w), w);
#endif
	}
}

void life_sim() {
#ifdef CUDA
	life_sim_row<<<HEIGHT,1>>>(tile[currBuffer], tile[1-currBuffer], width);
#else
	int y = height;
	while(--y) {
		life_sim_row(tile[currBuffer], tile[1-currBuffer], width, y);
	}
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