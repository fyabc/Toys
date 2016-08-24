#ifndef MY_UTILS_H
#define MY_UTILS_H

/* Print some boring information. */
void myPrint(int value);

/* Compute the greatest common divisor */
int gcd(int x, int y);

/* Test if (x0, y0) is in the Mandelbrot set or not */
int inMandel(double x0, double y0, int n);

/* Divide and modular two numbers */
int divMod(int a, int b, int* r);

/* Average values in an array */
double avg(double* a, int n);

/* A C data structure */
typedef struct Point {
	double x, y;
} Point;

/* Function involving a C data structure */
double distance(Point* p1, Point* p2);

#endif // MY_UTILS_H

