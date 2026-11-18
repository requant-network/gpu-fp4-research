// Original CUDA SHA-256 implementation from the equations/constants in RFC 6234.
// One independent digest per byte fragment; no floating-point operations.
__device__ __forceinline__ unsigned int rr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}
__constant__ unsigned int sha_k[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
    0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
    0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
    0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
    0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
    0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};
extern "C" __global__ void compression_sha256(
        const unsigned char* input, unsigned long long bytes, unsigned int fragment_bytes,
        unsigned char* output, unsigned long long fragments) {
    const unsigned long long id = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= fragments) return;
    const unsigned long long begin = id * fragment_bytes;
    const unsigned long long length = bytes - begin < fragment_bytes ? bytes - begin : fragment_bytes;
    const unsigned long long padded = ((length + 9 + 63) / 64) * 64;
    const unsigned long long bits = length * 8;
    unsigned int state[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                             0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    for (unsigned long long block = 0; block < padded; block += 64) {
        unsigned int w[64];
        for (unsigned int j = 0; j < 16; ++j) {
            unsigned int word = 0;
            for (unsigned int b = 0; b < 4; ++b) {
                const unsigned long long p = block + 4 * j + b;
                unsigned int value = 0;
                if (p < length) value = input[begin + p];
                else if (p == length) value = 0x80;
                else if (p >= padded - 8) value = (unsigned int)((bits >> (8 * (padded - 1 - p))) & 255);
                word = (word << 8) | value;
            }
            w[j] = word;
        }
        for (unsigned int j = 16; j < 64; ++j) {
            const unsigned int x = w[j-15], y = w[j-2];
            w[j] = w[j-16] + (rr(x,7)^rr(x,18)^(x>>3)) + w[j-7] + (rr(y,17)^rr(y,19)^(y>>10));
        }
        unsigned int a=state[0],b=state[1],c=state[2],d=state[3];
        unsigned int e=state[4],f=state[5],g=state[6],h=state[7];
        for (unsigned int j = 0; j < 64; ++j) {
            const unsigned int t1 = h + (rr(e,6)^rr(e,11)^rr(e,25)) + ((e&f)^((~e)&g)) + sha_k[j] + w[j];
            const unsigned int t2 = (rr(a,2)^rr(a,13)^rr(a,22)) + ((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
        state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    }
    for (unsigned int j = 0; j < 8; ++j)
        for (unsigned int b = 0; b < 4; ++b)
            output[id * 32 + j * 4 + b] = (unsigned char)(state[j] >> (24 - 8*b));
}
