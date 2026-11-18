// Original XBP1 CUDA decoder. Header/table bounds are checked again on device.
struct XbpHeader {
    unsigned char magic[4]; unsigned char version,itemsize,log,flags;
    unsigned long long words,blocks,bytes;
};
struct XbpEntry { unsigned long long base; unsigned int offset,code; };
static_assert(sizeof(XbpHeader)==32 && sizeof(XbpEntry)==16,"XBP1 CUDA layout");
extern "C" __global__ void xbp_decode(
        const unsigned char* encoded,unsigned long long encoded_bytes,void* output,
        unsigned long long expected_words,unsigned int expected_itemsize,unsigned int expected_log,
        unsigned int expected_flags,unsigned long long expected_blocks,unsigned int* status) {
    if(encoded_bytes<32){if(threadIdx.x==0)atomicOr(status,1u);return;}
    const XbpHeader h=*reinterpret_cast<const XbpHeader*>(encoded);
    if(h.magic[0]!='X'||h.magic[1]!='B'||h.magic[2]!='P'||h.magic[3]!='1'||h.version!=1||
       h.itemsize!=expected_itemsize||(h.itemsize!=4&&h.itemsize!=8)||h.log!=expected_log||
       h.log<4||h.log>12||h.flags!=expected_flags||h.flags>1||h.words!=expected_words||
       h.words>0xffffffffull||h.blocks!=expected_blocks||h.bytes!=encoded_bytes){
        if(threadIdx.x==0)atomicOr(status,1u);return;
    }
    if(h.flags==1){
        if(h.blocks!=0||encoded_bytes!=32+h.words*h.itemsize){if(threadIdx.x==0)atomicOr(status,1u);return;}
        const unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i<h.words){
            if(h.itemsize==4)reinterpret_cast<unsigned int*>(output)[i]=reinterpret_cast<const unsigned int*>(encoded+32)[i];
            else reinterpret_cast<unsigned long long*>(output)[i]=reinterpret_cast<const unsigned long long*>(encoded+32)[i];
        }
        return;
    }
    const unsigned long long block_words=1ull<<h.log;
    if(!h.words||h.blocks!=(h.words+block_words-1)/block_words){if(threadIdx.x==0)atomicOr(status,1u);return;}
    const unsigned long long payload_begin=32+h.blocks*16;
    if(payload_begin>encoded_bytes||(encoded_bytes-payload_begin)%8){if(threadIdx.x==0)atomicOr(status,1u);return;}
    if((unsigned long long)blockIdx.x>=h.blocks)return;
    const XbpEntry e=reinterpret_cast<const XbpEntry*>(encoded+32)[blockIdx.x];
    const unsigned int width=e.code&255,count=(e.code>>8)&65535,mode=e.code>>24;
    const unsigned long long begin=(unsigned long long)blockIdx.x*block_words;
    const unsigned long long remaining=h.words-begin;
    const unsigned int expected_count=remaining<block_words?(unsigned int)remaining:(unsigned int)block_words;
    const unsigned long long raw_words=((unsigned long long)count*h.itemsize+7)/8;
    const unsigned long long needed=mode?raw_words:((unsigned long long)count*width+63)/64;
    if(count!=expected_count||mode>1||(h.itemsize==4&&(e.base>>32))||
       (!mode&&(width>=h.itemsize*8||needed>=raw_words))||
       (mode&&(width!=h.itemsize*8||e.base!=0))){if(threadIdx.x==0)atomicOr(status,2u);return;}
    const unsigned long long available=(encoded_bytes-payload_begin)/8;
    if((unsigned long long)e.offset>available||needed>available-e.offset){if(threadIdx.x==0)atomicOr(status,4u);return;}
    const unsigned char* bytes=encoded+payload_begin+(unsigned long long)e.offset*8;
    const unsigned long long* packed=reinterpret_cast<const unsigned long long*>(bytes);
    if(threadIdx.x==0){
        if(!mode&&needed&&((unsigned long long)count*width)%64&&
           (packed[needed-1]>>(((unsigned long long)count*width)%64)))atomicOr(status,8u);
        if(mode)for(unsigned long long p=(unsigned long long)count*h.itemsize;p<needed*8;++p)
            if(bytes[p])atomicOr(status,8u);
    }
    for(unsigned int j=threadIdx.x;j<count;j+=blockDim.x){
        unsigned long long value=e.base;
        if(mode){
            value=h.itemsize==4?reinterpret_cast<const unsigned int*>(bytes)[j]:
                               reinterpret_cast<const unsigned long long*>(bytes)[j];
        }else if(width){
            const unsigned long long bit=(unsigned long long)j*width;
            const unsigned int shift=bit&63;
            unsigned long long difference=packed[bit>>6]>>shift;
            if(shift&&width>64-shift)difference|=packed[(bit>>6)+1]<<(64-shift);
            difference&=(1ull<<width)-1;
            value^=difference;
        }
        if(h.itemsize==4)reinterpret_cast<unsigned int*>(output)[begin+j]=(unsigned int)value;
        else reinterpret_cast<unsigned long long*>(output)[begin+j]=value;
    }
}
