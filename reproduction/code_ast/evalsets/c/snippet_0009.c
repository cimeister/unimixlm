#include "xc.h"

void Button_Init(void) {
    // define RA4-0 is input
    TRISA = 0x1F;
    // define RB0 is input
    TRISBbits.RB0 = 1;
    // clear
    PORTA = 0x00;
    PORTB = 0x00;
}