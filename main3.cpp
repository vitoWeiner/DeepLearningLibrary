
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>

#include <raylib.h>
#include <raymath.h>
#include <opencv2/opencv.hpp>

extern "C" void addToData(float* p, int target);
extern "C" void initTrainingHWD();
extern "C" void train();
extern "C" void initData();
extern "C" void clearData();
extern "C" void eval();

static std::vector<float> loadGrayToVector(const std::string& path, int w, int h) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("Failed to load image: " + path);

    if (img.cols != w || img.rows != h) {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        img = resized;
    }

    std::vector<float> out;
    out.reserve(w * h);
    for (int y = 0; y < h; ++y) {
        const uchar* row = img.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) out.push_back(row[x] / 255.0f);
    }
    return out;
}


int main4(void) {


    std::vector<std::vector<float>> images_2;
    int w = 28;
    int h = 28;

    images_2.reserve(10);

    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/5.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/16.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/25.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/28.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/76.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/82.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/381.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/262.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/381.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/684.png", w, h));

    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/54676.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/54674.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/54653.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/54134.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/53340.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/53008.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/53266.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/53284.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/53799.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/55040.png", w, h));

    std::vector<std::vector<float>> images_3;

    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/7.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/10.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/12.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/27.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/44.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/49.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/874.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/50.png", w, h));

    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/34368.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/34873.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/34862.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/34853.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/34967.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/34931.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/35796.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/36263.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/36755.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/37167.png", w, h));

       

    initTrainingHWD();

    for (auto& v : images_2) {
        addToData(v.data(), 2);
    }

    for (auto& v : images_3) {
        addToData(v.data(), 3);
    }

    initData();

    train();

    clearData();

    images_3.clear();
    images_2.clear();


    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/59996.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/59980.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/59978.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/59957.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/59891.png", w, h));
    images_3.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/3/59883.png", w, h)); // ovaj je bio zadnji od 3
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/59991.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/59824.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/59667.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/59655.png", w, h));
    images_2.push_back(loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/MNIST_dataset/mnist_png/train/2/59556.png", w, h));

    for (auto& im : images_3) {
        addToData(im.data(), 3);
    }

    for (auto& im : images_2) {
        addToData(im.data(), 2);
    }

    eval();


    

	return 0;
}