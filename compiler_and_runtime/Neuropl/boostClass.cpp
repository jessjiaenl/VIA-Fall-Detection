#include "boostClass.h"

template <typename T>
Test<T> ::Test(std::string name) {
    std::cout << "constructor " << std::endl;
    message = name;
}

template <typename T>
T Test<T>::myMax(T x, T y){
    return (x > y) ? x : y;
}

template <typename T>
void Test<T>::scream(){
    std::cout << "ARGGGHHHHHHHHH" << std::endl;
}

template <typename T>
void Test<T>::printMsg(){
    std::cout << message << std::endl;
}

int main(){
    Test<int> test = Test<int>("hi");
    test.scream();
    std::cout << test.myMax(1, 5) << std::endl;
    test.printMsg();
}