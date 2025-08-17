
/*

#include <iostream>
#include <vector>
#include <string>
#include <raylib.h>
#include <opencv2/opencv.hpp>


extern "C" void initTraining(const float* target_data, int width, int height);
extern "C" void trainStep();
extern "C" void getCurrentOutput(float* out_buffer);
extern "C" float getCost();

std::vector<float> loadGrayscaleImageAsVector(const std::string& image_path, int& out_width, int& out_height) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Greška prilikom u?itavanja slike: " + image_path);
    }

    out_width = image.cols;
    out_height = image.rows;

    std::vector<float> data;
    data.reserve(out_width * out_height);

    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            data.push_back(( image.at<uchar>(y, x)) / 255.0f);
        }
    }

    return data;
}


int main() {

    srand(time(0));

    const int img_width = 28;
    const int img_height = 28;
    const int scale = 10;
    const int win_width = img_width * scale;
    const int win_height = img_height * scale;

    std::vector<float> target_image;
    try {
        target_image = loadGrayscaleImageAsVector(
            
            "C:/Users/Vito/Desktop/DesktopApp/DeepLearningLib/smile.png",
            //"C:/Users/Vito/Desktop/DesktopApp/DeepLearningLibrary/10057.png",
            
            const_cast<int&>(img_width), const_cast<int&>(img_height)
        );
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    initTraining(target_image.data(), img_width, img_height);


    InitWindow(win_width * 2 + 20, win_height, "Training Visualization");
    SetTargetFPS(15);

    Image img = GenImageColor(img_width, img_height, RAYWHITE);
    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            float value = target_image[y * img_width + x];
            unsigned char intensity = static_cast<unsigned char>(value * 255);
            ImageDrawPixel(&img, x, y, Color{ intensity, intensity, intensity, 255 });
        }
    }
    Texture2D target_texture = LoadTextureFromImage(img);
    UnloadImage(img);

    std::vector<float> current_output(img_width * img_height);

    while (!WindowShouldClose()) {

        trainStep();
        getCurrentOutput(current_output.data());
        //std::cout << getCost() << "\n";


        Color* pixels = new Color[img_width * img_height];
        for (int i = 0; i < img_width * img_height; ++i) {
            unsigned char intensity = static_cast<unsigned char>(current_output[i] * 255);
            pixels[i] = Color{ intensity, intensity, intensity, 255 };
        }

        Image output_img{
              pixels,        // data
              img_width,     // width
              img_height,    // height
              1,             // mipmaps
              PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 // format
        };

        Texture2D output_texture = LoadTextureFromImage(output_img);


        UnloadImage(output_img);

        // 3. Crtanje obje slike
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Lijevo: cilj
        DrawTextureEx(target_texture, Vector2{ 0, 0 }, 0.0f, (float)scale, WHITE);
        // Desno: trenutni izlaz
        DrawTextureEx(output_texture, Vector2{ (float)(img_width * scale + 10), 0 }, 0.0f, (float)scale, WHITE);

        EndDrawing();


        UnloadTexture(output_texture);
    }

    UnloadTexture(target_texture);
    CloseWindow();


    return 0;
}

*/