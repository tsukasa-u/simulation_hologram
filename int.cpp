#include <iostream>
#include "legendre.h"
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <complex>

#define T std::complex<double>

#define pix 8.5E-6
#define z1 (1500.0E-3/pix)
#define lam (632.8E-9/pix)
// #define pix (1.0E-3)
#define scale 1.0

#define xRange 1024
#define yRange 1024

#define imageWidth  1024
#define imageHeight 1024

#define xRangeH xRange
#define yRangeH yRange

#define imageWidthH imageWidth
#define imageHeightH imageHeight

#define PIXEL(i) (i - xRange/2.0 + 0.5)

#define n 16
#define k 17

#define aH 0.0
#define bH xRange

#define CH (bH - aH)/2.0

#define xH(t) (bH - aH)*t/2.0 + (bH + aH)/2.0

#define N 1024.0

double (*P)(double) = P16;
double *x_k = x_17;

double W[k];

cv::Mat dst(imageHeight, imageWidth, CV_64FC2, cv::Scalar(1.0, 0.0));
cv::Mat src(imageHeight, imageWidth, CV_64FC2, cv::Scalar(1.0, 0.0));
cv::Mat kxX(xRangeH, xRange, CV_64FC2, cv::Scalar(1.0, 0.0));
cv::Mat yKy(yRange, yRangeH, CV_64FC2, cv::Scalar(1.0, 0.0));

std::complex<double> c(0.0, 1.0);

T f(double x, double xp, double kp) {
    return std::exp(c*M_PI*(2.0*(kp/N*x)+(x-xp)*(x-xp)/z1/lam));
}

T integral(double xp, double kp) {
    T temp(0.0, 0.0);
    for (int i = 0; i < k; i++) {
        temp += W[i]*f(xH(x_k[i]), xp, kp);
    }
    return temp;
}

T r(double xt, double kt) {
    return std::exp(xt*kt/N*2.0*M_PI*c);
}

int main() {

    // std::cout << yKy << std::endl;
    // std::cout << dst(cv::Range(0,yRangeH), cv::Range(0,xRangeH)) << std::endl;
    // std::cout << kxX << std::endl;

    // cv::Mat res = yKy*dst(cv::Range(0,yRangeH), cv::Range(0,xRangeH))*kxX;
    
    {
        cv::Mat png = cv::imread("src.png");
        // std::cout << png << std::endl;
        std::vector<cv::Mat> planes;
        cv::split(png, planes);
        std::vector<cv::Mat> single;
        single.push_back(planes[0]);
        single.push_back(cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1));
	    cv::Mat dst;
        cv::merge(single, dst);
        dst.convertTo(src, CV_64FC2);
    }

    cv::dft(src, dst);

    // std::cout << dst(cv::Range(0,yRangeH), cv::Range(0,xRangeH)) << std::endl;

    for (int i = 0; i < k; i++) {
        W[i] = 2*(1 - x_k[i]*x_k[i])/(k*P(x_k[i]))/(k*P(x_k[i]));
    }

    {
        T temp;
        cv::Point2d *p = &kxX.at<cv::Point2d>(0, 0);
        for (int j = 0; j < imageWidthH; j++) {
            for (int i = 0; i < xRange; i++) {
                temp = integral(i, j);
                // temp = r(i, j);
                *p = cv::Point2d(temp.real(), temp.imag());
                p++;
            }
        }

        p = &yKy.at<cv::Point2d>(0, 0);
        for (int i = 0; i < yRange; i++) {
            for (int j = 0; j < imageHeightH; j++) {
                temp = integral(i, j);
                // temp = r(i, j);
                *p = cv::Point2d(temp.real(), temp.imag());
                p++;
            }
        }
    }

    cv::Mat res = yKy*dst(cv::Range(0,imageHeightH), cv::Range(0,imageWidthH))*kxX;
    // std::cout << yKy.dims << " " << yKy.channels() << " " << yKy.cols << yKy.rows << std::endl;
    // std::cout << dst.dims << " " << dst.channels() << " " << dst.cols << dst.rows << std::endl;
    // std::cout << kxX.dims << " " << kxX.channels() << " " << kxX.cols << kxX.rows << std::endl;
    // std::cout << res.dims << " " << res.channels() << " " << res.cols << res.rows << std::endl;

    {
        cv::Mat data;
        std::vector<cv::Mat> planes;
        // cv::Mat img(yRange, xRange, CV_8UC3);

        cv::split(res, planes);
        
        double max = 0;
        cv::minMaxLoc(planes[0], NULL, &max, NULL, NULL);
        planes[0].convertTo(data, CV_8UC1, 256.0/max);
        // int from_to[] = {0, 0, 0, 1, 0, 2};
        // cv::mixChannels(&data, 1, &img, 1, from_to, 3);

        std::vector<int> param;
        cv::imwrite("untitled.png", data, param);
    }


    // std::cout << planes[0] << std::endl;
    // std::cout << data << std::endl;
    // std::cout << img << std::endl;


    
    // {
        cv::Mat data;
        std::vector<cv::Mat> planes;
        cv::split(dst, planes);
        
        double max = 0;
        cv::minMaxLoc(planes[0], NULL, &max, NULL, NULL);
        planes[0].convertTo(data, CV_8UC1, 256.0/max);
        std::vector<int> param;
        cv::imwrite("dst.png", data, param);
    // }
}