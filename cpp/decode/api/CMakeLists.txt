add_executable(api_tester
    main.cc api_test.cc wav_list_compare_test.cc content_audit.cc ctc_endpoint_diag.cc
    pronounce_score_diag.cc stability_test.cc recognition_score_test.cc
    ../../benchmark/recognize.cc ../../cli/wav_input.cc ../../feature/audio_reader/wavreader.cc)
target_include_directories(api_tester PRIVATE . ../../third_party/json/single_include)
target_compile_definitions(api_tester PRIVATE
    MERCURY_DEFAULT_MODEL_DIR="${MERCURY_TEST_MODEL_DIR}"
    MERCURY_SMOKE_WAV="${PROJECT_SOURCE_DIR}/test/resource/StevJobs_Standford/processing/SteveJobsStanford00001.wav")
target_link_libraries(api_tester PRIVATE asr_api Threads::Threads)
set_target_properties(api_tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${MERCURY_OUTPUT_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")

add_test(NAME api_smoke COMMAND api_tester smoke "${MERCURY_TEST_MODEL_DIR}")
add_test(NAME stability_test COMMAND api_tester stability "${MERCURY_TEST_MODEL_DIR}")
add_test(NAME recognition_score_test COMMAND api_tester score "${MERCURY_TEST_MODEL_DIR}")
set_tests_properties(stability_test PROPERTIES TIMEOUT 60)
