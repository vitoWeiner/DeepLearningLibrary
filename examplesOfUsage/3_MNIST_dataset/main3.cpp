

/*

MNIST dataset from : https://www.kaggle.com/datasets/hojjatk/mnist-dataset

*/


#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>

#include <raylib.h>
#include <raymath.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

extern "C" void addToData(float* p, int target);
extern "C" void initTrainingHWD();
extern "C" void train();
extern "C" void initData();
extern "C" void clearData();
extern "C" void eval();
extern "C" int predictDigit(float* p);


static std::vector<std::vector<float>> loadImagesFromFolder(
    const std::string& folder,
    int w, int h,
    size_t limit = 300,
    bool shuffle = false)
{
    namespace fs = std::filesystem;

    std::vector<std::string> files;

    for (auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
                files.push_back(entry.path().generic_string());
            }
        }
    }

#if 0
    if (shuffle) {
        std::random_shuffle(files.begin(), files.end());
    }
#endif

    if (files.size() > limit) {
        files.resize(limit);
    }

    std::vector<std::vector<float>> out;
    out.reserve(files.size());

    for (auto& path : files) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        if (img.cols != w || img.rows != h) {
            cv::resize(img, img, cv::Size(w, h), 0, 0, cv::INTER_AREA);
        }

        std::vector<float> vec;
        vec.reserve(w * h);
        for (int y = 0; y < h; ++y) {
            const uchar* row = img.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) {
                vec.push_back(row[x] / 255.0f);
            }
        }

        out.push_back(std::move(vec));
    }

    return out;
}


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


int main2(void) {


    int w = 28;
    int h = 28;
    
    std::vector<std::vector<float>> images_0 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/0", w, h);
    std::vector<std::vector<float>> images_1 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/1", w, h);
    std::vector<std::vector<float>> images_2 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/2", w, h);
    std::vector<std::vector<float>> images_3 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/3", w, h);
    std::vector<std::vector<float>> images_4 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/4", w, h);
    std::vector<std::vector<float>> images_5 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/5", w, h);
    std::vector<std::vector<float>> images_6 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/6", w, h);
    std::vector<std::vector<float>> images_7 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/7", w, h);
    std::vector<std::vector<float>> images_8 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/8", w, h);
    std::vector<std::vector<float>> images_9 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/train/9", w, h);


    



    initTrainingHWD();

    for (auto& v : images_0) {
        addToData(v.data(), 0);
    }

    for (auto& v : images_1) {
        addToData(v.data(), 1);
    }

    for (auto& v : images_2) {
        addToData(v.data(), 2);
    }

    for (auto& v : images_3) {
        addToData(v.data(), 3);
    }
    for (auto& v : images_4) {
        addToData(v.data(), 4);
    }
    for (auto& v : images_5) {
        addToData(v.data(), 5);
    }
    for (auto& v : images_6) {
        addToData(v.data(), 6);
    }
    for (auto& v : images_7) {
        addToData(v.data(), 7);
    }
    for (auto& v : images_8) {
        addToData(v.data(), 8);
    }
    for (auto& v : images_9) {
        addToData(v.data(), 9);
    }

    initData();

    train();

    clearData();

    images_0.clear();
    images_1.clear();
   images_2.clear();
   images_3.clear();
   images_4.clear();
   images_5.clear();
   images_7.clear();
   images_8.clear();
   images_9.clear();

    images_0 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/0", 28, 28);
    images_1 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/1", 28, 28);
    images_2 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/2", 28, 28);
  images_3 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/3", 28, 28);
  images_4 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/4", 28, 28);
  images_5 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/5", 28, 28);
  images_6 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/6", 28, 28);
  images_7 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/7", 28, 28);
  images_8 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/8", 28, 28);
  images_9 = loadImagesFromFolder("C:/Users/Vito/Desktop/DesktopApp/MNIST_dataset/mnist_png/test/9", 28, 28);

    for (auto& im : images_3) {
        addToData(im.data(), 3);
    }

    for (auto& im : images_2) {
        addToData(im.data(), 2);
    }

    for (auto& im : images_4) {
        addToData(im.data(), 4);
    }


    eval();


    std::vector<float> image = loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/numero3.png", w, h);
    std::vector<float> image2 = loadGrayToVector("C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/numo2.png", w, h);
    int x = predictDigit(image.data());
    int y = predictDigit(image2.data());



    printf("\n\nprediction for 3:\n\n%d", x);
    printf("\n\nprediction for 2:\n\n%d", y);


    // raylib

    const int screenWidth = 280;
    const int screenHeight = 280;

    InitWindow(screenWidth, screenHeight, "Draw a digit (0-9)");
    SetTargetFPS(60);

    std::vector<float> canvasPixels(w * h, 0.0f);

    bool mouseDown = false;

    std::vector<std::vector<float>> canvasBuffer(screenHeight, std::vector<float>(screenWidth, 0.0f));

    int predictedNumber = -1;

    while (!WindowShouldClose()) {

        BeginDrawing();
        ClearBackground(BLACK);

        DrawText("Draw digit and press R to run prediction", 10, 10, 20, DARKGRAY);

    
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            int mx = GetMouseX();
            int my = GetMouseY();

          
            int radius = 12;
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int px = mx + dx;
                    int py = my + dy;
                    if (px >= 0 && px < screenWidth && py >= 0 && py < screenHeight) {
                        if (dx * dx + dy * dy <= radius * radius) {
                            canvasBuffer[py][px] = 1.0f;
                        }
                    }
                }
            }
        }

    
        for (int y = 0; y < screenHeight; y++) {
            for (int x = 0; x < screenWidth; x++) {
                if (canvasBuffer[y][x] > 0.0f) {
                    DrawPixel(x, y, WHITE);
                }
            }
        }

        if (IsKeyPressed(KEY_R)) {
            std::vector<float> input28x28(w * h, 0.0f);

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float sum = 0.0f;
                    for (int dy = 0; dy < screenHeight / h; dy++) {
                        for (int dx = 0; dx < screenWidth / w; dx++) {
                            int sx = x * (screenWidth / w) + dx;
                            int sy = y * (screenHeight / h) + dy;
                            sum += canvasBuffer[sy][sx];
                        }
                    }
                    input28x28[y * w + x] = sum / ((screenWidth / w) * (screenHeight / h));
                }
            }

            predictedNumber = predictDigit(input28x28.data());
        }

        if (IsKeyPressed(KEY_C)) {
            for (int i = 0; i < canvasBuffer.size(); ++i) {
                for (int j = 0; j < canvasBuffer[i].size(); ++j) {
                    canvasBuffer[i][j] = 0.0f;
                }
            }

            predictedNumber = -1;

        }

        if (predictedNumber > 0) {
            DrawText(TextFormat("Prediction: %d", predictedNumber), 10, 40, 30, RED);
        }


        EndDrawing();
    }

    CloseWindow();


  

    

	return 0;
}