#include "catch.hpp"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/kernels/load_store_fp16_data.h"

using namespace smaug;

void fillFp16Data(std::vector<float16>& data) {
    for (int i = 0; i < data.size(); i++)
        data[i] = fp16(i * 0.1);
}

void fillFp32Data(std::vector<float>& data) {
    for (int i = 0; i < data.size(); i++)
        data[i] = i * 0.1;
}

void verifyFp16Data(const std::vector<float16>& data) {
    for (int i = 0; i < data.size(); i++)
        REQUIRE(Approx(fp32(data[i])).epsilon(kEpsilon) == i * 0.1);
}

void verifyFp32Data(const std::vector<float>& data) {
    for (int i = 0; i < data.size(); i++)
        REQUIRE(Approx(data[i]).epsilon(kEpsilon) == i * 0.1);
}

void doFp16LoadTest(int numElems) {
    std::vector<float16> fp16Data(numElems, 0);
    std::vector<float> fp32Data(numElems, 0);
    fillFp16Data(fp16Data);
    host_load_fp16(fp32Data.data(), fp16Data.data(), numElems, 0, 0);
    verifyFp32Data(fp32Data);
}

void doFp16StoreTest(int numElems) {
    std::vector<float16> fp16Data(numElems, 0);
    std::vector<float> fp32Data(numElems, 0);
    fillFp32Data(fp32Data);
    host_store_fp16(fp32Data.data(), fp16Data.data(), numElems, 0, 0);
    verifyFp16Data(fp16Data);
}

TEST_CASE_METHOD(SmaugTest, "float16 to float32 convert/load", "[smvfp16]") {
    SECTION("Transfer size smaller than 4K page") { doFp16LoadTest(192); }
    SECTION("Transfer size 4K page") { doFp16LoadTest(2048); }
    SECTION("Transfer size larger than 4K page") { doFp16LoadTest(4800); }
}

TEST_CASE_METHOD(SmaugTest, "float32 to float16 convert/store", "[smvfp16]") {
    SECTION("Transfer size smaller than 4K page") { doFp16StoreTest(192); }
    SECTION("Transfer size 4K page") { doFp16StoreTest(2048); }
    SECTION("Transfer size larger than 4K page") { doFp16StoreTest(4800); }
}
