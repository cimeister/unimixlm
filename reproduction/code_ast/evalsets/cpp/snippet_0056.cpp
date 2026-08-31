#include "myrandom.h"

MyRandom::MyRandom(int seed)
{
    if(seed != -1)
    {
        qsrand(uint(seed));
    }
    else
    {
        qsrand(uint(QTime::currentTime().msec()));
    }
}

int MyRandom::get(int max)
{
    return qrand()%max;
}

double MyRandom::getFloat()
{
    return (qrand()%1001)/1000.0;
}