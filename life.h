#ifndef _LIFE_H_
#define _LIFE_H_

class Life {
    public:
/* prepare the board. Memory consumption is width*height bits.
 * width - map width in pixels
 * height - map height in pixels 
 */
Life(int width, int height);
~Life();

/* 
 * load contents of buffer with size into tile memory at offset.
 * +--------------+
 * |              |
 * |     +---+    |
 * |     +   + height
 * |off_y+---+    |
 * |      width   |
 * +-----+--------+
 *  off_x
 *
 * buf - pix seed buffer
 * width - npixels (bit) width of seed buffer
 * height - npixel height of seed buffer
 * off_x - offset X in pixels/bits
 * off_y - offset Y in pixels/bits
 */
void load(unsigned char *buf, int width, int height, int off_x, int off_y);

/*
 * execute one simulation step. 
 * returns number of live cells.
 */
void sim();

int val_at(int x, int y);

/* return a pointer to the current tile */
void *buffer();

private:
int width, height;
};

#endif