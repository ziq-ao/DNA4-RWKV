#include "BitIoStream.hpp"
#include <stdexcept>

BitInputStream::BitInputStream(const uint8_t* buffer, size_t len) :
    data(buffer), size(len), byteIndex(0), currentByte(0), numBitsRemaining(0) {}

int BitInputStream::read() {
    if (numBitsRemaining == 0) {
        if (byteIndex >= size) return -1; // EOF
        currentByte = data[byteIndex++];
        numBitsRemaining = 8;
    }
    numBitsRemaining--;
    return (currentByte >> numBitsRemaining) & 1;
}

int BitInputStream::readNoEof() {
    int result = read();
    if (result != -1)
        return result;
    else
        throw std::runtime_error("End of stream");
}

BitOutputStream::BitOutputStream(std::vector<uint8_t>& buf) :
    buffer(buf), currentByte(0), numBitsFilled(0) {}

void BitOutputStream::write(int b) {
    currentByte = (currentByte << 1) | b;
    numBitsFilled++;
    if (numBitsFilled == 8) {
        buffer.push_back(static_cast<uint8_t>(currentByte));
        currentByte = 0;
        numBitsFilled = 0;
    }
}

void BitOutputStream::finish() {
    while (numBitsFilled != 0) write(0);
}