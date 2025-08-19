

/*
source of idea for this example : https://youtu.be/I_3d83cvByY?si=DNVYcFz7Zwei9Lrv
*/

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>

#include <raylib.h>
#include <raymath.h>
#include <opencv2/opencv.hpp>


/*

This current example (4_training_image) is inspired by the project nn.h (link: https://github.com/tsoding/nn.h) by [Alexey Kutepov].

*/


extern "C" void initTraining(const float* img1_data, const float* img2_data, int width, int height);
extern "C" void trainStep();
extern "C" void getCurrentOutput(float* out_buffer, float flag, int width, int height);
extern "C" float getCost();
extern "C" void setOutputSwap(bool swap);

template<typename T>
static inline T clampv(T v, T lo, T hi) { return (v < lo) ? lo : (v > hi ? hi : v); }


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

int main_4() {  
    std::srand((unsigned)std::time(nullptr));

    const int imgW = 28;
    const int imgH = 28;
    const int scale = 10;
    const int winW = imgW * scale * 3 + 40;
    const int winH = imgH * scale + 40;

    std::vector<float> img1, img2;
    try {
        img1 = loadGrayToVector("C:/path/man.png", imgW, imgH);
        img2 = loadGrayToVector("C:/path/house.png", imgW, imgH);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }

    initTraining(img1.data(), img2.data(), imgW, imgH);

    InitWindow(winW, winH, "Training Visualization with Flag");
    SetTargetFPS(30);

    auto texFromFloatVector = [&](const std::vector<float>& v) -> Texture2D {
        Image im = GenImageColor(imgW, imgH, RAYWHITE);
        for (int y = 0; y < imgH; ++y) {
            for (int x = 0; x < imgW; ++x) {
                float f = clampv(v[y * imgW + x], 0.0f, 1.0f);
                unsigned char g = (unsigned char)(f * 255.0f);
                ImageDrawPixel(&im, x, y, { g, g, g, 255 });
            }
        }
        Texture2D t = LoadTextureFromImage(im);
        UnloadImage(im);
        SetTextureFilter(t, TEXTURE_FILTER_BILINEAR);
        return t;
        };

    Texture2D texImg1 = texFromFloatVector(img1);
    Texture2D texImg2 = texFromFloatVector(img2);

    std::vector<float> nnOut(imgW * imgH, 0.0f);

  
    std::vector<Color> pixels(imgW * imgH);
    Image outImageHeader = { pixels.data(), imgW, imgH, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
    Texture2D outTex = LoadTextureFromImage(outImageHeader);
    SetTextureFilter(outTex, TEXTURE_FILTER_BILINEAR);
   

    float flag = -1.0f;
    bool mappingSwap = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_S)) {
            mappingSwap = !mappingSwap;
            setOutputSwap(mappingSwap);
            std::cout << "Mapping swap = " << (mappingSwap ? "ON" : "OFF") << '\n';
        }

        if (IsKeyDown(KEY_LEFT))  flag -= 0.02f;
        if (IsKeyDown(KEY_RIGHT)) flag += 0.02f;
        flag = clampv(flag, -1.0f, 1.0f);

        trainStep();
        getCurrentOutput(nnOut.data(), flag, imgW, imgH);

        
        for (int i = 0; i < imgW * imgH; ++i) {
            float v = clampv(nnOut[i], 0.0f, 1.0f);
            unsigned char g = (unsigned char)(v * 255.0f);
            pixels[i] = { g, g, g, 255 };
        }
        UpdateTexture(outTex, pixels.data());

        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawText(TextFormat("Flag: %.2f move with arrows <- ->(LEFT/RIGHT) from -1 to -0.8", flag), 10, 10, 20, BLACK);

        DrawTextureEx(texImg1, Vector2{ 0, 40 }, 0.0f, (float)scale, WHITE);
        DrawTextureEx(texImg2, Vector2{ (float)(imgW * scale + 10), 40 }, 0.0f, (float)scale, WHITE);
        DrawTextureEx(outTex, Vector2{ (float)(2 * imgW * scale + 20), 40 }, 0.0f, (float)scale, WHITE);

        EndDrawing();
    }

    UnloadTexture(outTex);
    UnloadTexture(texImg2);
    UnloadTexture(texImg1);
    CloseWindow();
    return 0;
}
