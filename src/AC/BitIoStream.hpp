#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

class BitInputStream {
    const uint8_t* data;
    size_t size;
    size_t byteIndex;
    int currentByte;
    int numBitsRemaining;
public:
    // 接收原生内存指针和大小，替代 std::istream
    BitInputStream(const uint8_t* buffer, size_t len);
    int read();
    int readNoEof();
};

class BitOutputStream {
    std::vector<uint8_t>& buffer;
    int currentByte;
    int numBitsFilled;
public:
    // 直接推入 vector 裸内存，替代 std::ostream
    explicit BitOutputStream(std::vector<uint8_t>& buf);
    void write(int b);
    void finish();
};