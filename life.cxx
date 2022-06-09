#undef CUDA
extern "C" {

#include "life.inl"

void life_init(int width_, int height_) {
	int bytes = (width_ + 2) * (height_ + 2) * sizeof(cell_t);
	width = width_;
	height = height_;
	tile[0] = (cell_t *)malloc(bytes);
	tile[1] = (cell_t *)malloc(bytes);
	local_tile = (cell_t *)malloc(bytes);
}

void life_deinit() {
	free(tile[0]);
	free(tile[1]);
	tile[0] = tile[1] = 0;
	width = height = 0;
}

void life_sim() {
	int x, y;
	for(y=0;y<height-1;y++) {
		for(x=0;x<width-1;x++) {
			life_sim_cell(tile[currBuffer], tile[1-currBuffer], width, x, y);
		}
	}
	memset(tile[currBuffer], 0, width);
	memset(tile[currBuffer] + (width + 1) * height, 0, width);
	memset(tile[currBuffer] + (width + 1) * (height-1), 0, width);
	currBuffer = 1-currBuffer;
}

void life_load(cell_t *buf, int w, int h, int off_x, int off_y) {
	int j;
	for(j=1;j<h;j++) {
		memcpy(tile[currBuffer] + ((off_y+j+1)*width) + off_x + 1, buf+(j*w), w);
	}
}


cell_t *life_buffer() {
	memcpy(local_tile, tile[currBuffer], ((width)*(height+1)+1)*sizeof(cell_t));
	return local_tile;
}

}
