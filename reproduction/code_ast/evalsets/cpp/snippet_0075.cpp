#include <iostream>
//#include <sys/types.h>
#include <inttypes.h>
using namespace std;

#define PUZZLE_DIM 6

char color_name[PUZZLE_DIM][7] = {"purple",
                                  "red   ",
                                  "orange",
                                  "green ",
                                  "blue  ",
                                  "yellow"};
uint8_t height[PUZZLE_DIM][PUZZLE_DIM] = {{5, 2, 1, 3, 4, 0},\
                                          {4, 3, 0, 5, 2, 1},\
                                          {1, 0, 3, 2, 5, 4},\
                                          {1, 5, 2, 4, 0, 3},\
                                          {2, 0, 4, 1, 3, 5},\
                                          {4, 3, 5, 0, 1, 2}};
bool    used        [PUZZLE_DIM][PUZZLE_DIM];
uint8_t sorted_color[PUZZLE_DIM][PUZZLE_DIM];
int     solution_count = 0;

void print_solution () {
    for (uint8_t y = 0; y < 12 * PUZZLE_DIM + 1; y++) cout << "*";
    cout << endl;
    for (uint8_t y = 0; y < PUZZLE_DIM; y++) {
        cout << "* ";
        for (int x = 0; x < PUZZLE_DIM; x++) {
            cout << "height: " << (uint32_t) height[y][x] << " * ";
        }
        cout << endl << "* ";
        for (int x = 0; x < PUZZLE_DIM; x++) {
            cout <<  color_name[sorted_color[y][x]] << "    * ";
        }
        cout << endl;
        for (int y = 0; y < 12 * PUZZLE_DIM + 1; y++) cout << "*";
        cout << endl;
    }
    cout << endl;
};

void find_next_piece(uint8_t x_pos, uint8_t y_pos)
{
    if (y_pos == PUZZLE_DIM) {
        solution_count++;
        //print_solution ();
    } else {
        uint8_t color, i, matches;
        for (color = 0; color < PUZZLE_DIM; color++)  {
            matches = 0;
            if (!used[color][height[y_pos][x_pos]]) {
                for (i = 0; i < y_pos; i++) {
                    if (color == sorted_color[i][x_pos]) matches++;
                }
                for (i = 0; i < x_pos; i++) {
                    if (color == sorted_color[y_pos][i]) matches++;
                }
                if (matches == 0) {
                    sorted_color[y_pos][x_pos] = color;
                    used[color][height[y_pos][x_pos]] = true;
                    find_next_piece((x_pos + 1) % PUZZLE_DIM,
                                    (y_pos + ((x_pos  + 1) / PUZZLE_DIM)));
                    used[color][height[y_pos][x_pos]] = false;
                }
            }
        }
    }
};

int main (void)
{
    for (uint8_t i = 0; i < PUZZLE_DIM; i++) {
        for (uint8_t j = 0; j < PUZZLE_DIM; j++) {
            used[i][j] = false;
        }
    }
    find_next_piece (0, 0);
    cout << "solutions: " << solution_count << std::endl;
    return (0);
};