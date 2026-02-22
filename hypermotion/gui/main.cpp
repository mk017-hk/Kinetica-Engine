#include "HyperMotionApp.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    hm::gui::HyperMotionApp app;

    if (!app.initialize(1400, 900)) {
        std::cerr << "Failed to initialize HyperMotion Studio" << std::endl;
        return EXIT_FAILURE;
    }

    app.run();
    app.shutdown();

    return EXIT_SUCCESS;
}
