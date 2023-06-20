#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

int disable(FILE* stream)
{
    int saved = dup(fileno(stream));

    int tmpfd = ::open("/dev/null", O_WRONLY);
    ::dup2(tmpfd, fileno(stream));
    ::close(tmpfd);

    return saved;
}

void restore(FILE* stream, int saved)
{
    dup2(saved, fileno(stream));
    close(saved);
}

int main(int argc, char* argv[])
{
    int fd = disable(stderr);

    std::cerr << "ERROR1: is not visible" << std::endl;

    restore(stderr, fd);

    std::cerr << "ERROR2: is visible" << std::endl;

    return 0;
}
