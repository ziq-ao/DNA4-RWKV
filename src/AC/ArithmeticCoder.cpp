/* 
 * Reference arithmetic coding
 * 
 * Copyright (c) Project Nayuki
 * MIT License. See readme file.
 * https://www.nayuki.io/page/reference-arithmetic-coding
 */

#include <limits>
#include <stdexcept>
#include "ArithmeticCoder.hpp"

using std::uint32_t;
using std::uint64_t;


ArithmeticCoderBase::ArithmeticCoderBase(int numBits) {
    if (!(1 <= numBits && numBits <= 63))
        throw std::domain_error("State size out of range");
    numStateBits = numBits;
    fullRange = static_cast<decltype(fullRange)>(1) << numStateBits;
    halfRange = fullRange >> 1;  // Non-zero
    quarterRange = halfRange >> 1;  // Can be zero
    minimumRange = quarterRange + 2;  // At least 2
    maximumTotal = std::min(std::numeric_limits<decltype(fullRange)>::max() / fullRange, minimumRange);
    stateMask = fullRange - 1;
    low = 0;
    high = stateMask;
}


ArithmeticCoderBase::~ArithmeticCoderBase() {}


void ArithmeticCoderBase::update(const std::uint32_t* cumulative, int cumul_size, uint32_t symbol) {
    // State check
    if (low >= high || (low & stateMask) != low || (high & stateMask) != high)
        throw std::logic_error("Assertion error: Low or high out of range");
    uint64_t range = high - low + 1;
    if (!(minimumRange <= range && range <= fullRange))
        throw std::logic_error("Assertion error: Range out of range");
    
    // 【修改点】：直接从原生指针读取
    uint32_t total = cumulative[cumul_size - 1]; 
    uint32_t symLow = cumulative[symbol];
    uint32_t symHigh = cumulative[symbol + 1];
    
    if (symLow == symHigh)
        throw std::invalid_argument("Symbol has zero frequency");
    if (total > maximumTotal)
        throw std::invalid_argument("Cannot code symbol because total is too large");
    
    // Update range
    uint64_t newLow  = low + symLow  * range / total;
    uint64_t newHigh = low + symHigh * range / total - 1;
    low = newLow;
    high = newHigh;
    
    // While low and high have the same top bit value, shift them out
    while (((low ^ high) & halfRange) == 0) {
        shift();
        low  = ((low  << 1) & stateMask);
        high = ((high << 1) & stateMask) | 1;
    }
    // While low's top two bits are 01 and high's are 10, delete the second highest bit of both
    while ((low & ~high & quarterRange) != 0) {
        underflow();
        low = (low << 1) ^ halfRange;
        high = ((high ^ halfRange) << 1) | halfRange | 1;
    }
}


ArithmeticDecoder::ArithmeticDecoder(int numBits, BitInputStream &in) :
        ArithmeticCoderBase(numBits),
        input(in),
        code(0) {
    for (int i = 0; i < numStateBits; i++)
        code = code << 1 | readCodeBit();
}


uint32_t ArithmeticDecoder::read(const std::uint32_t* cumulative, int cumul_size) {
    // Translate from coding range scale to frequency table scale
    uint32_t total = cumulative[cumul_size - 1];
    if (total > maximumTotal)
        throw std::invalid_argument("Cannot decode symbol because total is too large");
    uint64_t range = high - low + 1;
    uint64_t offset = code - low;
    uint64_t value = ((offset + 1) * total - 1) / range;
    if (value * range / total > offset)
        throw std::logic_error("Assertion error");
    if (value >= total)
        throw std::logic_error("Assertion error");
    
    // 【修改点】：基于 cumul_size 进行二分查找
    uint32_t start = 0;
    uint32_t end = cumul_size - 1; 
    while (end - start > 1) {
        uint32_t middle = (start + end) >> 1;
        if (cumulative[middle] > value)
            end = middle;
        else
            start = middle;
    }
    if (start + 1 != end)
        throw std::logic_error("Assertion error");
    
    uint32_t symbol = start;
    if (!(cumulative[symbol] * range / total <= offset && offset < cumulative[symbol + 1] * range / total))
        throw std::logic_error("Assertion error");
    
    // 【修改点】：传递指针和大小
    update(cumulative, cumul_size, symbol); 
    
    if (!(low <= code && code <= high))
        throw std::logic_error("Assertion error: Code out of range");
    return symbol;
}


void ArithmeticDecoder::shift() {
    code = ((code << 1) & stateMask) | readCodeBit();
}


void ArithmeticDecoder::underflow() {
    code = (code & halfRange) | ((code << 1) & (stateMask >> 1)) | readCodeBit();
}


int ArithmeticDecoder::readCodeBit() {
    int temp = input.read();
    if (temp == -1)
        temp = 0;
    return temp;
}


ArithmeticEncoder::ArithmeticEncoder(int numBits, BitOutputStream &out) :
    ArithmeticCoderBase(numBits),
    output(out),
    numUnderflow(0) {}


void ArithmeticEncoder::write(const std::uint32_t* cumulative, int cumul_size, uint32_t symbol) {
    update(cumulative, cumul_size, symbol);
}


void ArithmeticEncoder::finish() {
    output.write(1);
}


void ArithmeticEncoder::shift() {
    int bit = static_cast<int>(low >> (numStateBits - 1));
    output.write(bit);
    
    // Write out the saved underflow bits
    for (; numUnderflow > 0; numUnderflow--)
        output.write(bit ^ 1);
}


void ArithmeticEncoder::underflow() {
    if (numUnderflow == std::numeric_limits<decltype(numUnderflow)>::max())
        throw std::overflow_error("Maximum underflow reached");
    numUnderflow++;
}