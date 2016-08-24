#include <stdio.h>
#include <math.h>
#include "my_utils.h"

void myPrint(int value) {
	switch (value) {
		case 1:
			printf("Hello, this is 1\n");
			break;
		case 2:
			printf("Ok, 2 now\n");
			break;
		default:
			printf("Hihi, default %d now\n", value);
			break;
	}
}

int gcd(int x, int y) {
	int g = y;
	while (x > 0) {
		g = x;
		x = y % x;
		y = g;
	}
	return g;
}

int inMandel(double x0, double y0, int n) {
	double x = 0, y = 0, xTemp;
	while (n > 0) {
		xTemp = x * x - y * y + x0;
		y = 2 * x * y + y0;
		x = xTemp;
		n -= 1;
		if (x * x + y * y > 4)
			return 0;
	}
	return 1;
}

int divMod(int a, int b, int* r) {
	int q = a / b;
	*r = a % b;
	return q;
}

double avg(double* a, int n) {
	int i;
	double total = 0.0;
	for (i = 0; i < n; ++i) {
		total += a[i];
	}
	return total / n;
}

double distance(Point* p1, Point* p2) {
	return hypot(p1->x - p2->x, p1->y - p2->y);
}
