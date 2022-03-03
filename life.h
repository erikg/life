#ifndef _LIFE_H_
#define _LIFE_H_

typedef unsigned char cell_t;

/* prepare the board. Memory consumption is width*height bits.
 * width - map width in pixels
 * height - map height in pixels 
 */
void life_init(int width, int height);
void life_deinit();

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
void life_load(cell_t *buf, int width, int height, int off_x, int off_y);

/*
 * execute one simulation step. 
 * returns number of live cells.
 */
void life_sim();

/* return a pointer to the current tile */
cell_t *life_buffer();

#endif
