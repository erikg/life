#ifndef _DISP_H_
#define _DISP_H_

void disp_init(int width, int height);
void disp_update(cell_t *buf, int width, int height);
void disp_swap();
int disp_input();
void disp_deinit();

#endif
