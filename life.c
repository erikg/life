#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <life.h>

#undef CUDA

#define val_at(_x, _y) (((_x)<0 || (_y)<0 || (_x)>(width-1) || (_y)>(width-1)) ? 0 : tile[currBuffer][(_x)+width*(_y)] ? 1 : 0)

#ifdef CUDA
__global__
#endif
void life_sim_row(unsigned char **tile, int currBuffer, int width, int y) {
	int x = width;
	while(--x) {
		unsigned char live = val_at(x-1, y) + val_at(x+1, y) + val_at(x-1, y-1) + val_at(x, y-1) \
							 + val_at(x+1, y-1) + val_at(x-1, y+1) + val_at(x, y+1) + val_at(x+1, y+1);
		unsigned char newlife = val_at(x,y)?((live>>1)&1):live==3;
		tile[1-currBuffer][x+width*y] = newlife?255:0;
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
#else
	free(tile[0]);
	free(tile[1]);
#endif
	tile[0] = tile[1] = 0;
	width = height = 0;
}

void life_load(unsigned char *buf, int w, int h, int off_x, int off_y) {
	int i,j;
	for(j=0;j<h;j++) {
#ifdef CUDA
		cudaMemcpy(
			buf+(j*w),
			tile[currBuffer] + ((off_y+j)*width) + off_x,
			w, cudaMemcpyHostToDevice);
#else
		memcpy(tile[currBuffer] + ((off_y+j)*width) + off_x, buf+(j*w), w);
#endif
	}
}

void life_sim() {
	int y = height, sum=0;
#ifdef CUDA
	while(--y) {
		life_sim_row<<<1,1>>>(tile, currBuffer, width, y);
	}
#else
	while(--y) {
		life_sim_row(tile, currBuffer, width, y);
	}
#endif
	currBuffer = 1-currBuffer;
}

void *life_buffer() {
#ifdef CUDA
	cudaMemcpy(local_tile, tile[currBuffer], width*height, cudaMemcpyDeviceToHost);
#else
	memcpy(local_tile, tile[currBuffer], width*height);
#endif
	return local_tile;
}
