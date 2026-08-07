/* 
 * Reference arithmetic coding
 * 
 * Copyright (c) Project Nayuki
 * MIT License. See readme file.
 * https://www.nayuki.io/page/reference-arithmetic-coding
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>
#include "BitIoStream.hpp"


/* 
 * Provides the state and behaviors that arithmetic coding encoders and decoders share.
 */
class ArithmeticCoderBase {
    
    /*---- Configuration fields ----*/
    protected: int numStateBits;
    protected: std::uint64_t fullRange;
    protected: std::uint64_t halfRange;
    protected: std::uint64_t quarterRange;
    protected: std::uint64_t minimumRange;
    protected: std::uint64_t maximumTotal;
    protected: std::uint64_t stateMask;
    
    /*---- State fields ----*/
    protected: std::uint64_t low;
    protected: std::uint64_t high;
    
    /*---- Constructor ----*/
    public: explicit ArithmeticCoderBase(int numBits);
    public: virtual ~ArithmeticCoderBase() = 0;
    
    /*---- Methods ----*/
    protected: virtual void update(const std::uint32_t* cumulative, int cumul_size, std::uint32_t symbol);
    protected: virtual void shift() = 0;
    protected: virtual void underflow() = 0;
};


/* 
 * Reads from an arithmetic-coded bit stream and decodes symbols.
 */
class ArithmeticDecoder final : private ArithmeticCoderBase {
    
    /*---- Fields ----*/
    private: BitInputStream &input;
    private: std::uint64_t code;
    
    /*---- Constructor ----*/
    public: explicit ArithmeticDecoder(int numBits, BitInputStream &in);
    
    /*---- Methods ----*/
    public: std::uint32_t read(const std::uint32_t* cumulative, int cumul_size);
    
    protected: void shift() override;
    protected: void underflow() override;
    
    private: int readCodeBit();
};


/* 
 * Encodes symbols and writes to an arithmetic-coded bit stream.
 */
class ArithmeticEncoder final : private ArithmeticCoderBase {
    
    /*---- Fields ----*/
    private: BitOutputStream &output;
    private: unsigned long numUnderflow;
    
    /*---- Constructor ----*/
    public: explicit ArithmeticEncoder(int numBits, BitOutputStream &out);
    
    /*---- Methods ----*/
    public: void write(const std::uint32_t* cumulative, int cumul_size, std::uint32_t symbol);
    public: void finish();
    
    protected: void shift() override;
    protected: void underflow() override;
};