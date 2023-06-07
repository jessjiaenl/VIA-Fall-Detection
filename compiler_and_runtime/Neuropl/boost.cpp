#include <boost/python.hpp>
#include <stdio.h>

char const* greet()
{
   return "hello, world";
}

BOOST_PYTHON_MODULE(hello_ext)
{
    using namespace boost::python;
    def("greet", greet);
}

int main(){
    const char* hi = greet();
    printf("%s\n", hi);
}