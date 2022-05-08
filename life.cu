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


void life_deinit() {
	cudaFree(tile[0]);
	cudaFree(tile[1]);
	tile[0] = tile[1] = 0;
	width = height = 0;
}

void life_load(cell_t *buf, int w, int h, int off_x, int off_y) {
	int j;
	for(j=1;j<h;j++) {
		cudaMemcpy(tile[currBuffer] + ((off_y+j+1)*width) + off_x + 1, buf+(j*w), w, cudaMemcpyHostToDevice);
	}
}

cell_t *life_buffer() {
	cudaMemcpy(local_tile, tile[currBuffer], width*height*sizeof(cell_t), cudaMemcpyDeviceToHost);
	return local_tile;
}

}
