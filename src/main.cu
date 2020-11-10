#include <iostream>
#include "run_tests.h"

int main(int argc, char* argv[]) {

    if(argc == 2){
        std::cout << sha256_on_cpu(argv[1]) << std::endl;
    }

    run_tests();
    return 0;
}
