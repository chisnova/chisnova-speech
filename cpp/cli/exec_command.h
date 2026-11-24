#pragma once

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

// Forward arguments verbatim and give the caller the child program's exit status.
inline int exec_command(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    execvp(argv.front(), argv.data());
    const int error = errno;
    std::cerr << args.front() << ": " << std::strerror(error) << '\n';
    return error == ENOENT ? 127 : 126;
}
