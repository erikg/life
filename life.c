#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <life.h>

unsigned char *tile[2];	// device
unsigned char *local_tile;	// host
unsigned char currBuffer = 0;	// device
int width, height;	// device


static unsigned char val_at(int x, int y) {
	if(x<0 || y<0 || x>(width-1) || y>(width-1)) return 0;
	return tile[currBuffer][x+width*y] ? 1 : 0;
}

int life_val_at(int x, int y) {
	return val_at(x,y) ? 255 : 0;
}

void life_init(int width_, int height_) {
	width = width_;
	height = height_;
	tile[0] = (unsigned char *)malloc(width * height * 4);
	tile[1] = (unsigned char *)malloc(width * height * 4);
	local_tile = (unsigned char *)malloc(width * height * 4);
}

void life_deinit() {
	free(tile[0]);
	free(tile[1]);
	tile[0] = tile[1] = 0;
	width = height = 0;
}

void life_load(unsigned char *buf, int w, int h, int off_x, int off_y) {
	int i,j;
	for(j=0;j<h;j++) {
		memcpy(tile[currBuffer] + ((off_y+j)*width) + off_x, buf+(j*w), w);
	}
}

void life_sim_row(int y) {
	int x = width;
	while(--x) {
		unsigned char live = val_at(x-1, y) + val_at(x+1, y) + val_at(x-1, y-1) + val_at(x, y-1) \
							 + val_at(x+1, y-1) + val_at(x-1, y+1) + val_at(x, y+1) + val_at(x+1, y+1);
		unsigned char newlife = val_at(x,y)?((live>>1)&1):live==3;
		tile[1-currBuffer][x+width*y] = newlife?255:0;
	}
}

void life_sim() {
	int y = height, sum=0;
	while(--y) {
		life_sim_row(y);
	}
	currBuffer = 1-currBuffer;
}

void *life_buffer() {
	return memcpy(local_tile, tile[currBuffer], width*height);
}
