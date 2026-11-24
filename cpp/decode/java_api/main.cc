#include "cli/exec_command.h"
#include <filesystem>
#include <string_view>

int main(int argc, char** argv) {
    if (argc == 1 || argc > 3 || std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: java_api_test MODEL_DIR WAV_LIST\n       java_api_test SAMPLE_COUNT\n";
        return argc == 2 ? 0 : 2;
    }
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) {
        std::cerr << "Cannot locate java_api_test: " << error.message() << '\n';
        return 1;
    }
    const auto dir = executable.parent_path();
    std::vector<std::string> args{MERCURY_JAVA, "-Djava.library.path=" + dir.string(),
        "-Dmercury.test.data=" MERCURY_JAVA_DATA, "-Dmercury.test.model=" MERCURY_JAVA_MODEL,
        "-jar", (dir / MERCURY_TEST_JAR).string()};
    args.insert(args.end(), argv + 1, argv + argc);
    return exec_command(std::move(args));
}
