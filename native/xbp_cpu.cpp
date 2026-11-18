// Original XBP1 CPU encoder/validator. No ndzip implementation is used here.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
struct Header {
    char magic[4]; uint8_t version, itemsize, log, flags;
    uint64_t words, blocks, bytes;
};
struct Entry { uint64_t base; uint32_t offset, code; };
static_assert(sizeof(Header) == 32 && sizeof(Entry) == 16, "XBP1 layout");
thread_local std::string error;
void require(bool good, const char* message) { if (!good) throw std::runtime_error(message); }
uint32_t log_for(uint32_t block) {
    require(block >= 16 && block <= 4096 && !(block & (block-1)), "Invalid XBP1 block size");
    uint32_t log = 0; while (block >>= 1) ++log; return log;
}
uint64_t bound_for(uint64_t n, uint32_t itemsize, uint32_t block) {
    log_for(block);
    require((itemsize == 4 || itemsize == 8) && n <= UINT32_MAX, "Unsupported XBP1 type/count");
    return 32 + ((n + block - 1) / block) * 16 + ((n * itemsize + 7) / 8) * 8;
}
Entry read_entry(const unsigned char* p, uint64_t i) {
    Entry e; std::memcpy(&e, p + 32 + i*16, 16); return e;
}
void store_entry(unsigned char* p, uint64_t i, const Entry& e) { std::memcpy(p + 32 + i*16, &e, 16); }
template<class T> uint64_t word_at(const unsigned char* p, uint64_t i) {
    T value; std::memcpy(&value, p + i*sizeof(T), sizeof(T)); return value;
}
template<class T> uint64_t encode(const unsigned char* input, uint64_t n, uint32_t block,
                                 uint32_t threads, unsigned char* output) {
    const uint64_t blocks = (n + block - 1) / block;
    #pragma omp parallel for schedule(static) num_threads(threads) if(blocks > 1)
    for (int64_t index = 0; index < static_cast<int64_t>(blocks); ++index) {
        const uint64_t begin = uint64_t(index) * block, count = std::min<uint64_t>(block, n-begin);
        const uint64_t first = word_at<T>(input, begin);
        uint64_t differences = 0;
        for (uint64_t j = 0; j < count; ++j) differences |= word_at<T>(input, begin+j) ^ first;
        const uint32_t width = differences ? 64u - __builtin_clzll(differences) : 0;
        const uint32_t packed = (count * width + 63) / 64, raw = (count * sizeof(T) + 7) / 8;
        const bool fallback = packed >= raw;
        Entry e{fallback ? 0 : first, fallback ? raw : packed,
                (fallback ? uint32_t(sizeof(T)*8) : width) | (uint32_t(count)<<8) | (uint32_t(fallback)<<24)};
        store_entry(output, index, e);
    }
    uint64_t payload_words = 0;
    for (uint64_t index = 0; index < blocks; ++index) {
        Entry e = read_entry(output, index); const uint64_t length = e.offset;
        require(payload_words <= UINT32_MAX, "XBP1 payload offset limit");
        e.offset = static_cast<uint32_t>(payload_words); store_entry(output, index, e);
        payload_words += length;
    }
    const uint64_t payload_begin = 32 + blocks * 16;
    const uint64_t packed_total = payload_begin + payload_words*8, raw_total = 32 + n*sizeof(T);
    Header header{{'X','B','P','1'}, 1, uint8_t(sizeof(T)), uint8_t(log_for(block)), 0, n, blocks, packed_total};
    if (packed_total >= raw_total) {
        header.flags = 1; header.blocks = 0; header.bytes = raw_total;
        std::memcpy(output, &header, 32);
        if (n) std::memcpy(output+32, input, n*sizeof(T));
        return raw_total;
    }
    std::memcpy(output, &header, 32);
    #pragma omp parallel for schedule(static) num_threads(threads) if(blocks > 1)
    for (int64_t index = 0; index < static_cast<int64_t>(blocks); ++index) {
        const Entry e = read_entry(output, index);
        const uint32_t width = e.code & 255, count = (e.code >> 8) & 65535, mode = e.code >> 24;
        const uint64_t begin = uint64_t(index)*block;
        const uint64_t length = mode ? (uint64_t(count)*sizeof(T)+7)/8 : (uint64_t(count)*width+63)/64;
        unsigned char* destination = output + payload_begin + uint64_t(e.offset)*8;
        if (length) std::memset(destination, 0, length*8);
        if (mode) {
            std::memcpy(destination, input + begin*sizeof(T), uint64_t(count)*sizeof(T));
        } else if (width) {
            auto* packed = reinterpret_cast<uint64_t*>(destination);
            for (uint32_t j = 0; j < count; ++j) {
                const uint64_t value = word_at<T>(input, begin+j) ^ e.base;
                const uint64_t bit = uint64_t(j)*width; const uint32_t shift = bit & 63;
                packed[bit >> 6] |= value << shift;
                if (shift && width > 64-shift) packed[(bit >> 6)+1] |= value >> (64-shift);
            }
        }
    }
    return packed_total;
}
void validate(const unsigned char* input, uint64_t bytes, uint64_t* meta) {
    require(input && bytes >= 32, "Truncated XBP1 header");
    Header h; std::memcpy(&h, input, 32);
    require(!std::memcmp(h.magic,"XBP1",4) && h.version==1, "Unknown XBP1 magic/version");
    require((h.itemsize==4 || h.itemsize==8) && h.log>=4 && h.log<=12 && h.flags<=1, "Invalid XBP1 type/block/mode");
    require(h.words<=UINT32_MAX && h.bytes==bytes, "Invalid native XBP1 count/length");
    const uint64_t block = uint64_t(1)<<h.log;
    uint64_t payload_begin = 32;
    if (h.flags) {
        require(h.blocks==0 && bytes==32+h.words*h.itemsize, "Invalid XBP1 raw extent");
    } else {
        require(h.words && h.blocks==(h.words+block-1)/block, "Invalid XBP1 block count");
        payload_begin = 32+h.blocks*16;
        require(payload_begin<=bytes && (bytes-payload_begin)%8==0, "Truncated XBP1 table/payload");
        require(bytes<32+h.words*h.itemsize, "Non-beneficial packed XBP1 frame");
        uint64_t cursor = 0;
        for (uint64_t i=0;i<h.blocks;++i) {
            const Entry e=read_entry(input,i);
            const uint32_t width=e.code&255,count=(e.code>>8)&65535,mode=e.code>>24;
            const uint64_t expected=std::min<uint64_t>(block,h.words-i*block),raw=(uint64_t(count)*h.itemsize+7)/8;
            require(count==expected && e.offset==cursor, "Invalid XBP1 block extent/offset");
            require(mode<=1 && (h.itemsize==8 || (e.base>>32)==0), "Invalid XBP1 block mode/base");
            uint64_t length;
            if (!mode) {
                length=(uint64_t(count)*width+63)/64;
                require(width<h.itemsize*8 && length<raw, "Invalid XBP1 packed width/extent");
            } else {
                length=raw;require(width==h.itemsize*8 && e.base==0, "Invalid XBP1 raw block");
            }
            require(cursor<=((bytes-payload_begin)/8) && length<=((bytes-payload_begin)/8)-cursor,
                    "XBP1 payload outside packet");
            const uint64_t start=payload_begin+cursor*8,stop=start+length*8;
            if (!mode && length && (uint64_t(count)*width)%64) {
                uint64_t last;std::memcpy(&last,input+stop-8,8);
                require((last>>((uint64_t(count)*width)%64))==0,"Nonzero XBP1 unused packed bits");
            }
            if (mode) for(uint64_t p=start+uint64_t(count)*h.itemsize;p<stop;++p)
                require(input[p]==0,"Nonzero XBP1 raw padding");
            cursor+=length;
        }
        require(payload_begin+cursor*8==bytes,"Trailing XBP1 payload words");
    }
    const uint64_t values[8]={h.words,h.itemsize,h.log,h.flags,h.blocks,h.bytes,payload_begin,block};
    std::memcpy(meta,values,sizeof(values));
}
}
extern "C" {
const char* xbp_error(){return error.c_str();}
uint64_t xbp_bound(uint64_t n,uint32_t itemsize,uint32_t block){
    try {error.clear();return bound_for(n,itemsize,block);}catch(const std::exception& e){error=e.what();return 0;}
}
int xbp_encode(const void* data,uint64_t n,uint32_t itemsize,uint32_t block,uint32_t threads,
               void* output,uint64_t capacity,uint64_t* written){
    try {
        error.clear();const uint16_t endian=1;
        require(*reinterpret_cast<const uint8_t*>(&endian)==1,"Native XBP1 requires little-endian host");
        require((data || !n) && output && written && threads>=1 && threads<=64,"Invalid XBP1 encoding argument");
        require(reinterpret_cast<uintptr_t>(output)%8==0,"XBP1 candidate buffer must be 8-byte aligned");
        require(capacity>=bound_for(n,itemsize,block),"XBP1 candidate buffer too small");
        *written=itemsize==4?encode<uint32_t>(static_cast<const unsigned char*>(data),n,block,threads,
                                            static_cast<unsigned char*>(output)):
                               encode<uint64_t>(static_cast<const unsigned char*>(data),n,block,threads,
                                            static_cast<unsigned char*>(output));
        return 0;
    }catch(const std::exception& e){error=e.what();return 1;}
}
int xbp_validate(const void* input,uint64_t bytes,uint64_t* meta){
    try {error.clear();require(meta,"Missing XBP1 metadata result");validate(static_cast<const unsigned char*>(input),bytes,meta);return 0;}
    catch(const std::exception& e){error=e.what();return 1;}
}
}
