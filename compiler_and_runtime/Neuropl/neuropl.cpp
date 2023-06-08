#include "neuropl.h"
#include <boost/python.hpp>
/* Constructor function. */
/* Specify both loop_count and num_results. */

/* Using default arguements for loop_count and num_results*/
template <typename T>
Neuropl<T>::Neuropl(std::string path){ 
    std::cout << "constructor " << std::endl;
    model_path = path;
    setup();
}

/* Function implementations. */

/* This function should set up all the needed enviroments for the Neuron Runtime. */
template <typename T>
void Neuropl<T>::setup(){
    std::cout << "setup " << std::endl;
    /* Call all the NeuroRuntime functions just like how runtime.cpp does. */
}

/* These function is for testing purposes only. */
template <typename T>
void Neuropl<T>::print_attributes(void){
    std::cout << "print_attributes " << std::endl;
    std::cout << "model_path: " + model_path << std::endl;

}

template <typename T>
void Neuropl<T>::setModelPath(std::string path){
    std::cout << "SetModelPath " << std::endl;
    model_path = path;

}

//T predict(uint8_t* image) {
template <typename T>
T Neuropl<T>::predict(cv::Mat& image) {    
    //uint8_t* ptr = image.data;
    std::cout << "predict " << std::endl;
    return T {};
}

//template <typename T>
//T Neuropl<T>::predict(std::vector& image) {    
    //uint8_t* ptr = image.data;
//    return T {};
//}

int main(void){
    std::cout << "Using 2 default arguments" << std::endl;
    //neuropl *n = new neuropl("I scream :O");
    typedef std::vector<std::vector<uint8_t>> outfmt;

    std::string model_path {"model.dla"};
    
    /* 2 ways to call a function in C++. */
    Neuropl<outfmt> model{model_path};
    Neuropl<outfmt> m2 = Neuropl<outfmt>(model_path);
    model.print_attributes();
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


BOOST_PYTHON_MODULE(neuropl)
{
    using namespace boost::python;
    
    class_<Neuropl<uint8_t>>("Neuropl",  init<std::string>())
         .def("predict", &Neuropl<uint8_t>::predict)
         .def("print_attributes", &Neuropl<uint8_t>::print_attributes)
         .def("setModelPath", &Neuropl<uint8_t>::setModelPath)
        ;
}