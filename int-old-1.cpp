#include <iostream>
#include "legendre.h"
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <complex>

#define T std::complex<double>

#define z1 1500.0E-3
#define lam 632.8E-9

#define xRange 1024
#define yRange 1024

#define xRangeH xRange/2
#define yRangeH yRange/2

#define pix 8.5E-6
#define PIXEL(i) pix*(i - xRange/2.0 + 0.5)

#define n 16
#define k 17

#define aV -yRange/2.0*pix
#define bV yRange/2.0*pix
#define aH -xRange/2.0*pix
#define bH xRange/2.0*pix

#define CV (bV - aV)/2.0
#define CH (bH - aH)/2.0

#define xV(t) (bV - aV)*t/2.0 + (bV + aV)/2.0
#define xH(t) (bH - aH)*t/2.0 + (bH + aH)/2.0

#define N 64

double (*P)(double) = P16;
double *x_k = x_17;

double W[k];

cv::Mat dst(xRange, yRange, CV_64FC2);
cv::Mat src(xRange, yRange, CV_64FC1, cv::Scalar(1.0));

T d[xRange*xRange];
cv::Mat data(xRange, yRange, CV_8UC3);

T result[xRange*yRange];

std::complex<double> c(0.0, 1.0);

T f(int x_i, int y_i, double x, double y, double xp, double yp) {
    std::complex<double> im(0.0, 0.0);
    double *p = &dst.at<double>(0, 0);
    for (int i = 0; i < xRangeH; i++) {
        for (int j = 0; j < xRangeH; j++) {
            im += (((*p)++)+c*((*p)++))*std::exp(2.0*c*M_PI*(((double)j)/N*x_i+((double)i)/N*y_i));
        }
    }
    return im*std::exp(c*M_PI*((x-xp)*(x-xp)+(y-yp)*(y-yp))/z1/lam);
}

T calc(double xp, double yp) {
    
    T sum = 0;
    T temp;

    for (int i = 0; i < k; i++) {
        temp = 0;
        for (int j = 0; j < k; j++) {
            temp += W[j]*f(j, i, xH(x_k[j]), xV(x_k[i]), xp, yp);
        }
        sum += W[i]*temp;
    }
    return sum*CH*CV;
}

int main() {

    cv::dft(src, dst);

    for (int i = 0; i < k; i++) {
        W[i] = 2*(1 - x_k[i]*x_k[i])/(k*P(x_k[i]))/(k*P(x_k[i]));
    }

    // // cv::Mat src = cv::imread("src.png");
    

    double max = 0;
    {
        T *p = d;
        for (int i = 0; i < yRange; i++) {
            for (int j = 0; j < xRange; j++) {
                *p = calc(PIXEL(i), PIXEL(i));
                if (max < p->real()) max = p->real();
                p++;
            }
        }
    }
    {
        T *p = d;
        uint8_t temp;
        cv::Vec3b *srcp;
        for (int i = 0; i < yRange; ++i) {
            srcp = data.ptr<cv::Vec3b>(i);
            for (int j = 0; j < xRange; ++j) {
                temp = (uint8_t) (p->real()*255.0/max);
                srcp[j] = cv::Vec3b(temp, temp, temp);
                p++;
            }
        }
    }

    std::vector<int> param;
    cv::imwrite("untitled.png", data, param);
}