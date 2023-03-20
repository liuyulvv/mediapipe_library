#ifndef MEDIAPIPE_LOG_H_
#define MEDIAPIPE_LOG_H_

#ifdef _WIN32
#ifndef LibraryExport
#define LibraryExport __declspec(dllexport)
#endif
#else
#ifndef LibraryExport
#define LibraryExport __attribute__((visibility("default")))
#endif
#endif

#include <string>

class LibraryExport MediapipeLogger {
public:
    MediapipeLogger() = default;
    virtual ~MediapipeLogger() = default;

    virtual void Log(const std::string& content) const = 0;
};

#endif