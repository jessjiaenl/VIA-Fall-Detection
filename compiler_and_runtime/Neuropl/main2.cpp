#include <cstdlib>

int main(void){
    std::cout << "Welcome" << std::endl;

    std::string model_path {"./../model1.dla"};
    Neuropl model {model_path, 1, 2};
    
    // C++ example
    cv::Mat image(224, 224, CV_8UC3);
    // Get the buffer pointer and size
    uint8_t* byte_buffer = reinterpret_cast<uint8_t*>(image.data);
    auto result = model.predict<uint8_t>(byte_buffer);

    for (auto v : result) {
        for (auto vv : v) {
            std::cout << vv << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}