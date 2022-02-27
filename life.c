#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <life.h>

unsigned char *tile[2];
unsigned char currBuffer = 0;
#define otherBuffer (1-currBuffer)
int width, height;

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
  tile[0] = malloc(width * height);
  tile[1] = malloc(width * height);
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
    memcpy(tile[currBuffer] + ((off_y+h)*width) + off_x, buf+(j*w), w);
  }
}

void life_sim() {
  int y = height, sum=0;
  while(--y) {
    int x = width;
    while(--x) {
      unsigned char live = val_at(x-1, y) + val_at(x+1, y) + val_at(x-1, y-1) + val_at(x, y-1) \
        + val_at(x+1, y-1) + val_at(x-1, y+1) + val_at(x, y+1) + val_at(x+1, y+1);
      unsigned char newlife = val_at(x,y)?((live>>1)&1):live==3;
      tile[otherBuffer][x+width*y] = newlife?255:0;
    }
  }
  currBuffer = otherBuffer;
}

void *life_buffer() {
  (void *)(tile+currBuffer);
}