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

#define ScreenWidth  1080
#define ScreenHeight 1080

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

cv::Mat dst(xRange, yRange, CV_64FC2, cv::Scalar(1.0, 0.0));
cv::Mat src(xRange, yRange, CV_64FC2, cv::Scalar(1.0, 0.0));
cv::Mat kxX(xRangeH, xRange, CV_64FC2, cv::Scalar(1.0, 0.0));
cv::Mat yKy(yRange, yRangeH, CV_64FC2, cv::Scalar(1.0, 0.0));
// cv::Mat kxX = cv::Mat_<T>(xRangeH, xRange);
// cv::Mat yKy = cv::Mat_<T>(yRange, yRangeH);

std::complex<double> c(0.0, 1.0);

T f(double x, double xp, double kp) {
    return std::exp(c*M_PI*(2.0*(kp/N*x)+(x-xp)*(x-xp)/z1/lam));
}

T integral(double xp, double kp) {
    T temp(0.0, 0.0);
    for (int i = 0; i < k; i++) {
        temp += W[i]*f(xH(x_k[i]), xp, kp);
    }
    return temp*CH;
}

int main() {

    // std::cout << yKy << std::endl;
    // std::cout << dst(cv::Range(0,yRangeH), cv::Range(0,xRangeH)) << std::endl;
    // std::cout << kxX << std::endl;

    // cv::Mat res = yKy*dst(cv::Range(0,yRangeH), cv::Range(0,xRangeH))*kxX;

    cv::dft(src, dst);

    for (int i = 0; i < k; i++) {
        W[i] = 2*(1 - x_k[i]*x_k[i])/(k*P(x_k[i]))/(k*P(x_k[i]));
    }

    {
        T temp;
        cv::Point2d *p = &kxX.at<cv::Point2d>(0, 0);
        for (int j = 0; j < xRangeH; j++) {
            for (int i = 0; i < xRange; i++) {
                temp = integral(PIXEL(i), j);
                *p = cv::Point2d(temp.real(), temp.imag());
                p++;
            }
        }

        p = &yKy.at<cv::Point2d>(0, 0);
        for (int i = 0; i < yRange; i++) {
            for (int j = 0; j < yRangeH; j++) {
                temp = integral(PIXEL(i), j);
                *p = cv::Point2d(temp.real(), temp.imag());
                p++;
            }
        }
    }

    cv::Mat res = yKy*dst(cv::Range(0,yRangeH), cv::Range(0,xRangeH))*kxX;
    // std::cout << yKy.dims << " " << yKy.channels() << " " << yKy.cols << yKy.rows << std::endl;
    // std::cout << dst.dims << " " << dst.channels() << " " << dst.cols << dst.rows << std::endl;
    // std::cout << kxX.dims << " " << kxX.channels() << " " << kxX.cols << kxX.rows << std::endl;
    // std::cout << res.dims << " " << res.channels() << " " << res.cols << res.rows << std::endl;

    std::vector<cv::Mat> planes;
    cv::split(res, planes);
    
    double max = 0;
    cv::minMaxLoc(planes[0], NULL, &max, NULL, NULL);
    cv::Mat data;
    planes[0].convertTo(data, CV_8UC1, 256.0/max);
    cv::Mat img(yRange, xRange, CV_8UC3);
    int from_to[] = {0, 0, 0, 1, 0, 2};
    cv::mixChannels(&data, 1, &img, 1, from_to, 3);

    // // // cv::Mat src = cv::imread("src.png");

    // std::cout << planes[0] << std::endl;
    // std::cout << data << std::endl;
    // std::cout << img << std::endl;

    std::vector<int> param;
    cv::imwrite("untitled.png", data, param);
}