#include <vector>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const int HIST_SIZE = 256;

void plot_hist(const std::vector<int>& hist, const char* name) {

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound(static_cast<double>(hist_w) / HIST_SIZE);
    cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    normalize(hist, hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for(int i = 1; i < HIST_SIZE; i++) {
        cv::line(hist_image,
             cv::Point(bin_w * (i - 1), hist_h - hist[i - 1]),
             cv::Point(bin_w * i,       hist_h - hist[i]),
             cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    cv::imshow(name, hist_image);
    cv::waitKey();
}


void extract_original_hist(const cv::Mat& img) {

    std::vector<int> hist(HIST_SIZE, 0);
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            ++hist[(int)img.at<uchar>(r, c)];

    //plot_hist(hist, "original histogram");
}


int calc_cdf_val(int val, const std::vector<int>& hist) {

    int prob = 0;
    for (int i = val; i >= 0; i--)
        prob += hist[i];

    return prob;
}

void hist_equalize(const cv::Mat& img) {

    std::vector<int> hist(HIST_SIZE, 0);
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            ++hist[(int)img.at<uchar>(r, c)];

    int min_cdf = HIST_SIZE;
    std::vector<int> cdf(HIST_SIZE, 0);
    for (int i = 0; i < HIST_SIZE; i++) {
        cdf[i] = calc_cdf_val(i, hist);
        if (cdf[i] != 0)
            min_cdf = std::min(min_cdf, cdf[i]);
    }
    plot_hist(cdf, "cdf histogram");
    // normalize CDF
    int m = (int)img.total() - 1;
    int l = HIST_SIZE;
    std::vector<int> h_cdf(HIST_SIZE, 0);
    for (int i = 0; i < HIST_SIZE; i++)
        h_cdf[i] = cvRound((double(cdf[i] - min_cdf) / m) * (l - 1));

    //

    cv::Mat norm_img(img.size(), img.type());
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            norm_img.at<uchar>(r, c) = h_cdf[(int)img.at<uchar>(r, c)];

    cv::imshow("normal", norm_img);
}


int main() {

    cv::Mat img = imread("../res/gray.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) return -1;
    cv::imshow("gray", img);
    cv::waitKey();

    extract_original_hist(img);
    hist_equalize(img);
    cv::waitKey();

    return 0;
}
