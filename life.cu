#define CUDA 1
extern "C" {

#include "life.inl"

void life_init(int width_, int height_) {
	int bytes = (width_ + 2) * (height_ + 2) * sizeof(cell_t);
	width = width_;
	height = height_;
	cudaMalloc(tile, bytes);
	cudaMalloc(tile+1, bytes);
	local_tile = (cell_t *)malloc(bytes);
}

void life_sim() {
	life_sim_cell<<<dim3(WIDTH,HEIGHT),1>>>(tile[currBuffer], tile[1-currBuffer], width);

	/*
	for(int i=0;i<height+2;i++) {
		cell_t *dst = tile[currBuffer]+ (i * width * sizeof(cell_t));
		dst[0].val = dst[width].val = 0;
	}
	*/
	currBuffer = 1-currBuffer;
}

}
