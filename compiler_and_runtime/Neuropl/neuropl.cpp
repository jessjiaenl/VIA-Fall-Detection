#include "neuropl.h"
#include "RuntimeAPI.h"

int main(void){
    neuropl *n = new neuropl("hi\n");
    n->print_path();

}