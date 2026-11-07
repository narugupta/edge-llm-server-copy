# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(llama-bench)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(server)
    endif()
    add_subdirectory(mtmd)
    if (GGML_RPC)
    endif()
    if (LLAMA_CONNECT)
        add_subdirectory(connect)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
    endif()
    if (GGML_METAL)
        add_subdirectory(tuning)
    endif()
endif()
