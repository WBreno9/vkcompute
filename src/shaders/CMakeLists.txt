set(GLSL_VALIDATOR glslangValidator)

set( SHADERS
    ${CMAKE_SOURCE_DIR}/src/shaders/sum.comp
)

foreach(GLSL ${SHADERS})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${PROJECT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/shaders/"
        COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
        DEPENDS ${GLSL}
    )
    list(APPEND SHADERS_BINARY ${SPIRV})
endforeach(GLSL)

add_custom_target(shaders DEPENDS ${SHADERS_BINARY})
