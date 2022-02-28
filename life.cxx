#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <life.h>

unsigned char *tile[2]; // front and back on device
unsigned char *localTile; // local copy
unsigned char currBuffer = 0;
#define otherBuffer (1-currBuffer)

static unsigned char life_val_at(int width, int x, int y) {
  if(x<0 || y<0 || x>(width-1) || y>(width-1)) return 0;
  return tile[currBuffer][x+width*y] ? 1 : 0;
}

int Life::val_at(int x, int y) {
  return life_val_at(width, x,y) ? 255 : 0;
}

Life::Life(int width, int height) {
	this->width = width;
	this->height = height;
  tile[0] = (unsigned char *)malloc(width * height);
  tile[1] = (unsigned char *)malloc(width * height);
  localTile = (unsigned char *)malloc(width * height);
}

Life::~Life() {
  free(tile[0]);
  free(tile[1]);
  free(localTile);
  localTile = tile[0] = tile[1] = NULL;
  width = height = 0;
}

void Life::load(unsigned char *buf, int w, int h, int off_x, int off_y) {
  int i,j;
  for(j=0;j<h;j++) {
    memcpy(tile[currBuffer] + ((off_y+j)*width) + off_x, buf+(j*w), w);
  }
}

static inline void sim_row(int y, int width) {
    int x = width;
    while(--x) {
      unsigned char live = life_val_at(width, x-1, y) + life_val_at(width, x+1, y) + life_val_at(width, x-1, y-1) + life_val_at(width, x, y-1) \
        + life_val_at(width, x+1, y-1) + life_val_at(width, x-1, y+1) + life_val_at(width, x, y+1) + life_val_at(width, x+1, y+1);
      unsigned char newlife = life_val_at(width, x,y)?((live>>1)&1):live==3;
      tile[otherBuffer][x+width*y] = newlife?255:0;
    }
}

void Life::sim() {
  int y = height, sum=0;
  while(--y) {
	  sim_row(y, width);
  }
  currBuffer = otherBuffer;
}

void *Life::buffer() {
  memcpy(localTile, tile+currBuffer, width*height*sizeof(unsigned char));
  return localTile;
}
