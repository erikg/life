#include <SDL.h>
#include <SDL_opengl.h>

#include <GL/gl.h>
#include <GL/glext.h>

#include "disp.h"
#include "life.h"

#define renderer ((SDL_Renderer *)(this->blob))

Display::Display(int width, int height) {
    this->width = width;
    this->height = height;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                           width, height, SDL_WINDOW_SHOWN);
    this->blob = (void *)SDL_CreateRenderer(win, -1, 0);
}

void Display::update(void *buf, Life *l) {
    for(int y=0;y<this->height;y++) {
        for (int x=0;x<this->width;x++) {
            SDL_SetRenderDrawColor(renderer, l->val_at(x,y), 0, 48, SDL_ALPHA_OPAQUE);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
}

int Display::input() {
    SDL_Event event;
    while(SDL_PollEvent(&event)) {
        if(event.type == SDL_QUIT || (event.type == SDL_KEYUP && event.key.keysym.scancode == SDLK_ESCAPE)) {
            return 0;
        }
    }
    return -1;
}

void Display::swap() {
    SDL_RenderPresent(renderer);
}

Display::~Display() {
    SDL_Quit();
}

