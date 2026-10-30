cmake_minimum_required(VERSION 3.20)

project(
    mc_options_engine
    VERSION 0.1.0
    DESCRIPTION "C++20 Monte Carlo derivatives pricing and risk engine"
    LANGUAGES CXX
)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

function(mc_enable_warnings target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    elseif(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    endif()
endfunction()

add_library(
    mcpricing
    src/forecasting/historical_gbm.cpp
    src/forecasting/price_history_csv.cpp
    src/forecasting/walk_forward_backtest.cpp
    src/instruments/european_call.cpp
    src/instruments/european_put.cpp
    src/instruments/arithmetic_asian_call.cpp
    src/models/black_scholes.cpp
    src/models/geometric_brownian_motion.cpp
    src/pricing/analytical_black_scholes.cpp
    src/pricing/greeks_engine.cpp
    src/pricing/monte_carlo_engine.cpp
    src/pricing/path_monte_carlo_engine.cpp
    src/statistics/running_statistics.cpp
    src/validation.cpp
)
target_compile_features(mcpricing PUBLIC cxx_std_20)
target_include_directories(
    mcpricing
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
mc_enable_warnings(mcpricing)
target_link_libraries(mcpricing PUBLIC Threads::Threads)

add_library(
    mccli
    src/cli/backtest_options.cpp
    src/cli/drift_options.cpp
    src/cli/forecast_options.cpp
    src/cli/price_options.cpp
    src/cli/volatility_options.cpp
)
target_compile_features(mccli PUBLIC cxx_std_20)
target_link_libraries(mccli PUBLIC mcpricing)
mc_enable_warnings(mccli)

add_executable(mcprice src/main.cpp)
target_compile_features(mcprice PRIVATE cxx_std_20)
target_link_libraries(mcprice PRIVATE mccli)
mc_enable_warnings(mcprice)

add_executable(mc_convergence benchmarks/convergence.cpp)
target_compile_features(mc_convergence PRIVATE cxx_std_20)
target_link_libraries(mc_convergence PRIVATE mcpricing)
mc_enable_warnings(mc_convergence)

add_executable(mc_scaling benchmarks/scaling.cpp)
target_compile_features(mc_scaling PRIVATE cxx_std_20)
target_link_libraries(mc_scaling PRIVATE mcpricing)
mc_enable_warnings(mc_scaling)

add_executable(mc_antithetic benchmarks/antithetic.cpp)
target_compile_features(mc_antithetic PRIVATE cxx_std_20)
target_link_libraries(mc_antithetic PRIVATE mcpricing)
mc_enable_warnings(mc_antithetic)

add_executable(mc_greeks_variance benchmarks/greeks_variance.cpp)
target_compile_features(mc_greeks_variance PRIVATE cxx_std_20)
target_link_libraries(mc_greeks_variance PRIVATE mcpricing)
mc_enable_warnings(mc_greeks_variance)

add_executable(mc_thread_overhead benchmarks/thread_overhead.cpp)
target_compile_features(mc_thread_overhead PRIVATE cxx_std_20)
target_link_libraries(mc_thread_overhead PRIVATE mcpricing)
mc_enable_warnings(mc_thread_overhead)

add_executable(mc_hot_loop benchmarks/hot_loop.cpp)
target_compile_features(mc_hot_loop PRIVATE cxx_std_20)
target_link_libraries(mc_hot_loop PRIVATE mcpricing)
mc_enable_warnings(mc_hot_loop)

add_executable(mc_final_benchmarks benchmarks/final_suite.cpp)
target_compile_features(mc_final_benchmarks PRIVATE cxx_std_20)
target_link_libraries(mc_final_benchmarks PRIVATE mcpricing)
mc_enable_warnings(mc_final_benchmarks)

include(CTest)

if(BUILD_TESTING)
    find_package(Catch2 3 QUIET)

    if(NOT Catch2_FOUND)
        include(FetchContent)
        if(POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()
        FetchContent_Declare(
            Catch2
            URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.15.3.tar.gz
        )
        FetchContent_MakeAvailable(Catch2)
    endif()

    add_executable(
        mc_tests
        tests/test_asian_option.cpp
        tests/test_backtest_cli.cpp
        tests/test_black_scholes.cpp
        tests/test_cli.cpp
        tests/test_domain_types.cpp
        tests/test_forecast_cli.cpp
        tests/test_forecasting.cpp
        tests/test_greeks.cpp
        tests/test_monte_carlo.cpp
        tests/test_multithreading.cpp
        tests/test_payoffs.cpp
        tests/test_path_simulation.cpp
        tests/test_price_history_csv.cpp
        tests/test_walk_forward_backtest.cpp
        tests/test_statistics.cpp
        tests/test_smoke.cpp
        tests/test_validation.cpp
    )
    target_compile_features(mc_tests PRIVATE cxx_std_20)
    target_link_libraries(mc_tests PRIVATE mcpricing mccli Catch2::Catch2WithMain)
    mc_enable_warnings(mc_tests)

    add_test(NAME mc_tests COMMAND mc_tests)
    find_package(Python3 3.9 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME energy_demand_baselines
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_energy_baselines.py)
        add_test(NAME neural_direction_models
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_neural_direction.py)
        add_test(NAME calibrated_direction_models
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_calibrated_direction.py)
        add_test(NAME stock_direction_models
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_direction_forecasts.py)
        add_test(NAME stock_feature_pipeline
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stock_features.py)
        add_test(NAME matched_forecast_runner
            COMMAND ${CMAKE_COMMAND} -E env MC_ENGINE=$<TARGET_FILE:mcprice>
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_matched_forecasts.py)
        add_test(NAME stock_history_downloader
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_download_stock_history.py)
        add_test(NAME greeks_cli_workflow
            COMMAND ${CMAKE_COMMAND} -E env MC_ENGINE=$<TARGET_FILE:mcprice>
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_greeks_cli.py)
        add_test(NAME forecast_benchmark_runner
            COMMAND ${CMAKE_COMMAND} -E env MC_ENGINE=$<TARGET_FILE:mcprice>
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_benchmark_forecasts.py)
        add_test(NAME forecast_residual_runner
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_residual_forecasts.py)
        add_test(NAME forecast_classical_runner
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_classical_forecasts.py)
        add_test(NAME forecast_boosted_runner
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_boosted_forecasts.py)
    endif()
    add_test(NAME history_cli_workflow
        COMMAND ${CMAKE_COMMAND}
            -DCLI=$<TARGET_FILE:mcprice>
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/history_cli_workflow.cmake)
endif()
