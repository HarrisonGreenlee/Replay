cmake_minimum_required(VERSION 3.15)
project(ContactNetwork LANGUAGES C)

# Use C11 standard
set(CMAKE_C_STANDARD 11)

# Find libxml2 via vcpkg
find_package(LibXml2 REQUIRED)

# Define executable target
add_executable(contact_network
    main.c
    graph_generator.c
)

# Include libxml2 headers
target_include_directories(contact_network PRIVATE ${LIBXML2_INCLUDE_DIR})

# Link libxml2 library
target_link_libraries(contact_network PRIVATE ${LIBXML2_LIBRARIES})

# Optional MSVC debug flags
if (MSVC)
    target_compile_options(contact_network PRIVATE /Zi /Od)
    target_link_options(contact_network PRIVATE /DEBUG)
endif()
