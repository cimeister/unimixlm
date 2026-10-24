#include <stdlib.h> 
#include <math.h>

double norma(int size, double * v)
{
    double sum = 0;
    for (int i = 0; i < size; ++i)
        sum += v[i] * v[i];

    return sqrt(sum);
}

void multiplyByScalar(int size, double * v, double scalar)
{
    for (int i = 0; i < size; ++i)
        v[i] *= scalar;
}

double * additionVectors(int size, double * v, double * w)
{
    double * x = malloc(size * sizeof(double));
    for (int i = 0; i < size; ++i)
        x[i] = v[i] + w[i];

    return x;
}

// Considering the vectors are represented in an orthogonal basis.
double scalarProduct(int size, double * v, double * w)
{
    double product = 0 ;
    for (int i = 0; i < size; ++i)
        product += v[i] * w[i];

    return product;
}

double * crossProduct(int size, double * v, double * w)
{
    double * x = malloc(size * sizeof(double));
    
    x[0] = v[1] * w[2] - w[1] * v[2];
    x[1] = -v[0] * w[2] + w[0] * v[2];
    x[2] = v[0] * w[1] - w[0] * v[1];

    return x;
}