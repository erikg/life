#include <SDL.h>
#include <SDL_opengl.h>

#include <GL/gl.h>
#include <GL/glext.h>

#include "life.h"
#include "disp.h"

SDL_Window *win = NULL;
SDL_Renderer *renderer = NULL;

void disp_init(int width, int height) {
    renderer = NULL;
    win = NULL;
    SDL_Init(SDL_INIT_VIDEO);
    win = SDL_CreateWindow("Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                           width+2, height+2, SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(win, -1, 0);
}

void disp_update(cell_t *buf, int width, int height) {
    int x,y;
    for(y=0;y<height;y++) {
        for (x=0;x<width;x++) {
	    cell_t val = buf[(y+1)*width+x+1];
            SDL_SetRenderDrawColor(renderer, val&0xf?255:0, val&0xf0?255:0, 48, SDL_ALPHA_OPAQUE);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
}

int disp_input() {
    SDL_Event event;
    while(SDL_PollEvent(&event)) {
        if(event.type == SDL_QUIT || (event.type == SDL_KEYUP && event.key.keysym.scancode == SDLK_ESCAPE)) {
            return 0;
        }
    }
    return -1;
}

void disp_swap() {
    SDL_RenderPresent(renderer);
}

void disp_deinit() {
    SDL_Quit();
}

