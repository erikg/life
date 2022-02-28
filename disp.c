#include <SDL.h>
#include <SDL_opengl.h>

#include <GL/gl.h>
#include <GL/glext.h>

#include "disp.h"
#include "life.h"

SDL_Window *win;
SDL_Renderer *renderer;

void disp_init(int width, int height) {
    SDL_Init(SDL_INIT_VIDEO);
    win = SDL_CreateWindow("Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                           width, height, SDL_WINDOW_SHOWN);
    renderer = SDL_CreateRenderer(win, -1, 0);
}

void disp_update(unsigned char *buf, int width, int height) {
    int x,y;
    for(y=0;y<height;y++) {
        for (x=0;x<width;x++) {
            SDL_SetRenderDrawColor(renderer, buf[y*width+x], 0, 48, SDL_ALPHA_OPAQUE);
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

