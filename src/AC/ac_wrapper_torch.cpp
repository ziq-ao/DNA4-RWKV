#include <torch/extension.h>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <stdexcept>
#include "ArithmeticCoder.hpp"
#include "BitIoStream.hpp"

// ==========================================
// 🚀 流式编码器类 (支持无限追加)
// ==========================================
class StreamingEncoder {
private:
    std::ofstream ofs;
    std::vector<uint8_t> buffer;
    std::unique_ptr<BitOutputStream> bos;
    std::unique_ptr<ArithmeticEncoder> enc;
    bool is_closed = false;

public:
    // 初始化：打开文件，建立内存流和编码器
    StreamingEncoder(const std::string& filepath) {
        ofs.open(filepath, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }
        
        // 预分配 1MB 的极速写入内存缓冲
        buffer.reserve(1024 * 1024); 
        bos = std::make_unique<BitOutputStream>(buffer);
        enc = std::make_unique<ArithmeticEncoder>(32, *bos);
    }

    // 循环编码：将当前 batch 加入流中
    void encode_batch(torch::Tensor cumuls, torch::Tensor symbols) {
        if (is_closed) throw std::runtime_error("Encoder is already closed!");

        int batch_size = symbols.size(0);
        int cumul_size = cumuls.size(1);
        const uint32_t* ptr_cumuls = reinterpret_cast<const uint32_t*>(cumuls.data_ptr<int32_t>());
        const uint32_t* ptr_symbols = reinterpret_cast<const uint32_t*>(symbols.data_ptr<int32_t>());

        for (int i = 0; i < batch_size; ++i) {
            const uint32_t* current_cumul_ptr = ptr_cumuls + i * cumul_size;
            enc->write(current_cumul_ptr, cumul_size, ptr_symbols[i]);
        }

        // 💡 内存保护与极限 I/O：如果缓冲超过 1MB，一次性写入硬盘，并清空缓冲
        // 这样可以实现无限追加编码，而内存始终保持在极低水平
        if (buffer.size() >= 1024 * 1024) {
            ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
            buffer.clear(); 
            // 注意：buffer.clear() 只是重置长度，不释放物理内存，
            // bos 里的状态 (未凑满 8 bit 的残存位) 也完全不受影响，完美无缝衔接！
        }
    }

    // 编码结束：输出收敛 bit，补齐尾部，写入文件并关闭
    void close() {
        if (is_closed) return;
        
        enc->finish();
        bos->finish();
        
        if (!buffer.empty()) {
            ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
            buffer.clear();
        }
        
        ofs.close();
        is_closed = true;
    }

    // 析构函数作为最后一道防线，防止忘记 close
    ~StreamingEncoder() {
        close();
    }
};

// ==========================================
// 🚀 流式解码器类
// ==========================================
class StreamingDecoder {
private:
    std::vector<uint8_t> in_buffer;
    std::unique_ptr<BitInputStream> bis;
    std::unique_ptr<ArithmeticDecoder> dec;

public:
    // 初始化：一次性把文件加载进内存，为连续解码做准备
    StreamingDecoder(const std::string& filepath) {
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) throw std::runtime_error("Failed to open file: " + filepath);
        
        std::streamsize file_size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        
        in_buffer.resize(file_size);
        if (!ifs.read(reinterpret_cast<char*>(in_buffer.data()), file_size)) {
            throw std::runtime_error("Failed to read input file!");
        }
        ifs.close();

        bis = std::make_unique<BitInputStream>(in_buffer.data(), in_buffer.size());
        dec = std::make_unique<ArithmeticDecoder>(32, *bis);
    }

    // 连续解码当前 batch
    torch::Tensor decode_batch(torch::Tensor cumuls) {
        int batch_size = cumuls.size(0);
        int cumul_size = cumuls.size(1);
        const uint32_t* ptr_cumuls = reinterpret_cast<const uint32_t*>(cumuls.data_ptr<int32_t>());

        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
        torch::Tensor symbols = torch::empty({batch_size}, options);
        uint32_t* ptr_symbols = reinterpret_cast<uint32_t*>(symbols.data_ptr<int32_t>());

        for (int i = 0; i < batch_size; ++i) {
            const uint32_t* current_cumul_ptr = ptr_cumuls + i * cumul_size;
            ptr_symbols[i] = dec->read(current_cumul_ptr, cumul_size);
        }

        return symbols;
    }
};

// ==========================================
// 🔗 Pybind11 绑定类
// ==========================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<StreamingEncoder>(m, "StreamingEncoder")
        .def(pybind11::init<const std::string&>())               // 构造函数
        .def("encode_batch", &StreamingEncoder::encode_batch)    // 循环追加
        .def("close", &StreamingEncoder::close);                 // 手动关闭

    pybind11::class_<StreamingDecoder>(m, "StreamingDecoder")
        .def(pybind11::init<const std::string&>())
        .def("decode_batch", &StreamingDecoder::decode_batch);
}