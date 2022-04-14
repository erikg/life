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
	life_sim_row<<<dim3(WIDTH,HEIGHT),1>>>(tile[currBuffer], tile[1-currBuffer], width);

	currBuffer = 1-currBuffer;
}

}