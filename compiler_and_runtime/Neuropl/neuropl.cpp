#include "neuropl.h"

/* Constructor function. */
/* Specify both loop_count and num_results. */

/* Using default arguements for loop_count and num_results*/
template <typename T>
Neuropl<T>::Neuropl(std::string& path){ 
    std::cout << "constructor " << std::endl;
    model_path = path;
}

/* Function implementations. */

/* This function should set up all the needed enviroments for the Neuron Runtime. */
template <typename T>
void Neuropl<T>::setup(){
    /* Call all the NeuroRuntime functions just like how runtime.cpp does*/
}

/* These function is for testing purposes only. */
template <typename T>
void Neuropl<T>::print_attributes(void){
    std::cout << "model_path: " + model_path << std::endl;

}

//T predict(uint8_t* image) {
template <typename T>
T Neuropl<T>::predict(cv::Mat& image) {    
    //uint8_t* ptr = image.data;

    return T {};
}

//template <typename T>
//T Neuropl<T>::predict(std::vector& image) {    
    //uint8_t* ptr = image.data;
//    return T {};
//}

int main(void){
    std::cout << "Using 2 default arguments\n" << std::endl;
    //neuropl *n = new neuropl("I scream :O");
    //n->print_attributes();
    typedef std::vector<std::vector<uint8_t>> outfmt;

    std::string model_path {"model.dla"};
    
    Neuropl<outfmt> model{model_path};
    //std::vector<uint8_t> output {10};

    //std::vector<std::vector<uint8_t>> output = model.predict(image); 
    //output.size;
    cv::Mat image {};

    outfmt output = model.predict(image);

    for (auto v : output) {
        for (auto vv : v) {
            std::cout << vv << " ";
        }
        std::cout << std::endl;
    }

}