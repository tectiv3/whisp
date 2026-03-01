cmake_minimum_required(VERSION 3.14)
project(whisp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Build whisper library only, skip examples/tests
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
add_subdirectory(whisper.cpp)

add_executable(whisp
    src/main.cpp
    whisper.cpp/examples/common.cpp
    whisper.cpp/examples/common-whisper.cpp
)

target_include_directories(whisp PRIVATE
    whisper.cpp/include
    whisper.cpp/examples
    whisper.cpp/examples/server
)

target_link_libraries(whisp PRIVATE
    whisper
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
