#include <iostream>
#include "legendre.h"
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <complex>

#define T std::complex<double>

#define z1 1500.0E-3
#define lam 632.8E-9

#define xRange 1080
#define yRange 1080

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

double (*P)(double) = P16;
double *x_k = x_17;

double W[k];

std::complex<double> c(0.0, 1.0);

double Constance(double x, double y) {
    return (x*x + y*y < pix*pix) ? 255.0 : 0.0;
}

double f(double x, double y, double xp, double yp) {
    // return pow(x, 12) - 15.0*pow(x, 2)*pow(y, 8) - 4.0*pow(x, 2)*pow(y, 3) + 2.0*x + 5.0*pow(y, 2);
    return x;
}

std::complex<double> fX(double x, double xp) {
    return std::exp(c*M_PI*(x-xp)*(x-xp)/z1/lam);
}

std::complex<double> fY(double y, double yp) {
    return std::exp(c*M_PI*(y-yp)*(y-yp)/z1/lam);
}

T d[xRange*xRange];
// uint8_t data[xRange*xRange];
cv::Mat data(xRange, yRange, CV_8UC3);

T resultX[xRange];
T resultY[yRange];

double calc(double xp, double yp) {
    
    double sum = 0;
    double temp;

    for (int i = 0; i < k; i++) {
        temp = 0;
        for (int j = 0; j < k; j++) {
            temp += W[j]*f(xH(x_k[j]), xV(x_k[i]), xp, yp);
        }
        sum += W[i]*temp;
    }
    return sum*CH*CV;
}

T calcX(double xp) {
    
    T sum = 0;

    for (int i = 0; i < k; i++) {
        sum += W[i]*fX(xH(x_k[i]), xp);
    }

    return sum*CH;
}

T calcY(double yp) {
    
    T sum = 0;

    for (int i = 0; i < k; i++) {
        sum += W[i]*fY(xV(x_k[i]), yp);
    }

    return sum*CV;
}

int main() {

    for (int i = 0; i < k; i++) {
        W[i] = 2*(1 - x_k[i]*x_k[i])/(k*P(x_k[i]))/(k*P(x_k[i]));
    }

    for (int i = 0; i < xRange; i++) {
        resultX[i] = calcX(PIXEL(i));
    }
        
    for (int i = 0; i < yRange; i++) {
        resultY[i] = calcY(PIXEL(i));
    }

    // cv::Mat src = cv::imread("src.png");
    

    double max = 0;
    {
        T temp;
        T *p = d;
        // cv::Vec3b *srcp;
        for (int i = 0; i < yRange; i++) {
            // srcp = src.ptr<cv::Vec3b>(i);
            temp = resultY[i];
            for (int j = 0; j < xRange; j++) {
                // std::cout << srcp[j] << std::endl;
                // *p = ((double)srcp[j][0])*temp*resultX[j];   // from image buffer
                *p = ((double)Constance(PIXEL(j), PIXEL(i)))*temp*resultX[j];
                // *p = temp*resultX[j];
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
                // std::cout << cv::Vec3b(temp, temp, temp);
                p++;
            }
        }
    }

    std::vector<int> param;
    cv::imwrite("untitled.png", data, param);
}