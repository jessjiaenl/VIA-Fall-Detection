#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct neuropl{
    const char *model_path;

    neuropl(const char *);
    void print_path();
};

//constructor

neuropl::neuropl(const char *path){
    model_path = path;
}

//function (for testing in main currently)

void neuropl::print_path(void){
    printf("path is = %s", model_path);
}