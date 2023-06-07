#include <boost/python.hpp>
#include <stdio.h>
#include <string>
#include <iostream>

template <typename T>
class Test{
    public:
        std::string message;
        Test(std::string name);
        T myMax(T x, T y); /* sample template T return type program. */
        void scream();
        void printMsg();

};

BOOST_PYTHON_MODULE(boostClass)
{
    using namespace boost::python;
    
    class_<Test<uint8_t>>("Test",  init<std::string>())
        .def("myMax", &Test<uint8_t>::myMax)
        .def("scream", &Test<uint8_t>::scream)
        .def("printMsg", &Test<uint8_t>::printMsg)
        ;
}