#include "commands.h"
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) try {
    using namespace api_tests;
    struct Command {
        const char* name;
        int (*run)(int, char**);
    };
    const Command commands[] = {
        {"smoke", smoke}, {"compare", compare}, {"audit", audit},
        {"endpoint", endpoint}, {"pronounce", pronounce}, {"score", score},
        {"stability", stability}, {"benchmark", benchmark},
    };

    for (const auto& cmd : commands) {
        if (argc > 1 && std::string_view(argv[1]) == cmd.name) return cmd.run(argc - 1, argv + 1);
    }
    if (argc == 3) return smoke(argc, argv);
    std::cerr << "Usage: api_tester MODEL_DIR WAV_LIST\n"
              << "        api_tester [smoke|compare|audit|endpoint|pronounce|score|stability|benchmark] [args...]\n";
    return argc == 2 && std::string_view(argv[1]) == "--help" ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << "api_tester: " << error.what() << '\n';
    return 1;
}
