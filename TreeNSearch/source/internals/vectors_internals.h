#pragma once
#include <array>
#include <vector>
#include <cstring>
#include <iostream>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// SIMD: real intrinsics on x86, scalar stand-ins on ARM / Apple Silicon
// ─────────────────────────────────────────────────────────────────────────────
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
  #if !defined(__AVX2__)
    #define __AVX2__
  #endif
#else
// ── Scalar stand-ins so the codebase compiles on Apple Silicon / ARM ─────────
#include <cstdint>

struct alignas(16) __m128 {
    float f[4];
};

// __m128i needs multiple views of the same 16 bytes
union alignas(16) __m128i {
    int8_t   c[16];
    int16_t  s[8];
    int32_t  i[4];
    int64_t  l[2];
    uint8_t  uc[16];
    uint16_t us[8];
    uint32_t ui[4];
};

// __m256 / __m256i
struct alignas(32) __m256 {
    float f[8];
};

union alignas(32) __m256i {
    int8_t   c[32];
    int16_t  s[16];
    int32_t  i[8];
    int64_t  l[4];
    uint8_t  uc[32];
    uint16_t us[16];
    uint32_t ui[8];
};

// ── __m128 float ops ──────────────────────────────────────────────────────────
static inline __m128  _mm_setzero_ps()                                { return __m128{}; }
static inline __m128  _mm_set1_ps(float v)                            { __m128 r{}; for(int i=0;i<4;i++) r.f[i]=v; return r; }
static inline __m128  _mm_load_ps(const float* p)                     { __m128 r{}; memcpy(&r,p,16); return r; }
static inline __m128  _mm_loadu_ps(const float* p)                    { __m128 r{}; memcpy(&r,p,16); return r; }
static inline void    _mm_store_ps(float* p, __m128 a)                { memcpy(p,&a,16); }
static inline void    _mm_storeu_ps(float* p, __m128 a)               { memcpy(p,&a,16); }
static inline __m128  _mm_add_ps(__m128 a, __m128 b)                  { __m128 r{}; for(int i=0;i<4;i++) r.f[i]=a.f[i]+b.f[i]; return r; }
static inline __m128  _mm_sub_ps(__m128 a, __m128 b)                  { __m128 r{}; for(int i=0;i<4;i++) r.f[i]=a.f[i]-b.f[i]; return r; }
static inline __m128  _mm_mul_ps(__m128 a, __m128 b)                  { __m128 r{}; for(int i=0;i<4;i++) r.f[i]=a.f[i]*b.f[i]; return r; }
static inline __m128  _mm_div_ps(__m128 a, __m128 b)                  { __m128 r{}; for(int i=0;i<4;i++) r.f[i]=a.f[i]/b.f[i]; return r; }
static inline __m128  _mm_min_ps(__m128 a, __m128 b)                  { __m128 r{}; for(int i=0;i<4;i++) r.f[i]=a.f[i]<b.f[i]?a.f[i]:b.f[i]; return r; }
static inline __m128  _mm_max_ps(__m128 a, __m128 b)                  { __m128 r{}; for(int i=0;i<4;i++) r.f[i]=a.f[i]>b.f[i]?a.f[i]:b.f[i]; return r; }
static inline __m128  _mm_and_ps(__m128 a, __m128 b)                  { __m128 r{}; uint32_t ai,bi,ri; for(int i=0;i<4;i++){memcpy(&ai,&a.f[i],4);memcpy(&bi,&b.f[i],4);ri=ai&bi;memcpy(&r.f[i],&ri,4);} return r; }
static inline __m128  _mm_or_ps (__m128 a, __m128 b)                  { __m128 r{}; uint32_t ai,bi,ri; for(int i=0;i<4;i++){memcpy(&ai,&a.f[i],4);memcpy(&bi,&b.f[i],4);ri=ai|bi;memcpy(&r.f[i],&ri,4);} return r; }
static inline __m128  _mm_andnot_ps(__m128 a, __m128 b)               { __m128 r{}; uint32_t ai,bi,ri; for(int i=0;i<4;i++){memcpy(&ai,&a.f[i],4);memcpy(&bi,&b.f[i],4);ri=(~ai)&bi;memcpy(&r.f[i],&ri,4);} return r; }
static inline __m128  _mm_cmplt_ps(__m128 a, __m128 b)                { __m128 r{}; uint32_t mask; for(int i=0;i<4;i++){mask=a.f[i]<b.f[i]?0xffffffff:0;memcpy(&r.f[i],&mask,4);} return r; }
static inline __m128  _mm_cmple_ps(__m128 a, __m128 b)                { __m128 r{}; uint32_t mask; for(int i=0;i<4;i++){mask=a.f[i]<=b.f[i]?0xffffffff:0;memcpy(&r.f[i],&mask,4);} return r; }
static inline __m128  _mm_cmpgt_ps(__m128 a, __m128 b)                { __m128 r{}; uint32_t mask; for(int i=0;i<4;i++){mask=a.f[i]>b.f[i]?0xffffffff:0;memcpy(&r.f[i],&mask,4);} return r; }
static inline __m128  _mm_cmpge_ps(__m128 a, __m128 b)                { __m128 r{}; uint32_t mask; for(int i=0;i<4;i++){mask=a.f[i]>=b.f[i]?0xffffffff:0;memcpy(&r.f[i],&mask,4);} return r; }
static inline int     _mm_movemask_ps(__m128 a)                        { int r=0; uint32_t m; for(int i=0;i<4;i++){memcpy(&m,&a.f[i],4);r|=((m>>31)&1)<<i;} return r; }
static inline __m128  _mm_blendv_ps(__m128 a, __m128 b, __m128 mask)  { __m128 r{}; uint32_t m; for(int i=0;i<4;i++){memcpy(&m,&mask.f[i],4);r.f[i]=(m>>31)?b.f[i]:a.f[i];} return r; }
static inline __m128i _mm_castps_si128(__m128 a)                       { __m128i r{}; memcpy(&r,&a,16); return r; }
static inline __m128  _mm_castsi128_ps(__m128i a)                      { __m128 r{}; memcpy(&r,&a,16); return r; }

// ── __m128i integer ops ───────────────────────────────────────────────────────
static inline __m128i _mm_setzero_si128()                              { return __m128i{}; }
static inline __m128i _mm_loadu_si128(const __m128i* p)                { __m128i r{}; memcpy(&r,p,16); return r; }
static inline void    _mm_storeu_si128(__m128i* p, __m128i a)          { memcpy(p,&a,16); }

// set1
static inline __m128i _mm_set1_epi8(int8_t v)                          { __m128i r{}; for(int i=0;i<16;i++) r.c[i]=v; return r; }
static inline __m128i _mm_set1_epi16(int16_t v)                        { __m128i r{}; for(int i=0;i<8;i++)  r.s[i]=v; return r; }
static inline __m128i _mm_set1_epi32(int32_t v)                        { __m128i r{}; for(int i=0;i<4;i++)  r.i[i]=v; return r; }
static inline __m128i _mm_set1_epi64x(int64_t v)                       { __m128i r{}; for(int i=0;i<2;i++)  r.l[i]=v; return r; }

// setr
static inline __m128i _mm_setr_epi8(int8_t a0,int8_t a1,int8_t a2,int8_t a3,int8_t a4,int8_t a5,int8_t a6,int8_t a7,
                                     int8_t a8,int8_t a9,int8_t a10,int8_t a11,int8_t a12,int8_t a13,int8_t a14,int8_t a15)
{ __m128i r{}; r.c[0]=a0;r.c[1]=a1;r.c[2]=a2;r.c[3]=a3;r.c[4]=a4;r.c[5]=a5;r.c[6]=a6;r.c[7]=a7;
               r.c[8]=a8;r.c[9]=a9;r.c[10]=a10;r.c[11]=a11;r.c[12]=a12;r.c[13]=a13;r.c[14]=a14;r.c[15]=a15; return r; }
static inline __m128i _mm_setr_epi16(int16_t a0,int16_t a1,int16_t a2,int16_t a3,int16_t a4,int16_t a5,int16_t a6,int16_t a7)
{ __m128i r{}; r.s[0]=a0;r.s[1]=a1;r.s[2]=a2;r.s[3]=a3;r.s[4]=a4;r.s[5]=a5;r.s[6]=a6;r.s[7]=a7; return r; }
static inline __m128i _mm_setr_epi32(int32_t a0,int32_t a1,int32_t a2,int32_t a3)
{ __m128i r{}; r.i[0]=a0;r.i[1]=a1;r.i[2]=a2;r.i[3]=a3; return r; }
static inline __m128i _mm_cvtsi64_si128(int64_t a)                     { __m128i r{}; r.l[0]=a; return r; }

// bitwise
static inline __m128i _mm_and_si128(__m128i a, __m128i b)              { __m128i r{}; for(int i=0;i<2;i++) r.l[i]=a.l[i]&b.l[i]; return r; }
static inline __m128i _mm_or_si128 (__m128i a, __m128i b)              { __m128i r{}; for(int i=0;i<2;i++) r.l[i]=a.l[i]|b.l[i]; return r; }
static inline __m128i _mm_xor_si128(__m128i a, __m128i b)              { __m128i r{}; for(int i=0;i<2;i++) r.l[i]=a.l[i]^b.l[i]; return r; }
static inline __m128i _mm_andnot_si128(__m128i a, __m128i b)           { __m128i r{}; for(int i=0;i<2;i++) r.l[i]=(~a.l[i])&b.l[i]; return r; }

// compare epi8
static inline __m128i _mm_cmpeq_epi8 (__m128i a, __m128i b)           { __m128i r{}; for(int i=0;i<16;i++) r.c[i]=(a.c[i]==b.c[i])?(int8_t)-1:0; return r; }

// compare epi16
static inline __m128i _mm_cmpeq_epi16(__m128i a, __m128i b)           { __m128i r{}; for(int i=0;i<8;i++) r.s[i]=(a.s[i]==b.s[i])?(int16_t)-1:0; return r; }
static inline __m128i _mm_cmplt_epi16(__m128i a, __m128i b)           { __m128i r{}; for(int i=0;i<8;i++) r.s[i]=(a.s[i]< b.s[i])?(int16_t)-1:0; return r; }
static inline __m128i _mm_cmpgt_epi16(__m128i a, __m128i b)           { __m128i r{}; for(int i=0;i<8;i++) r.s[i]=(a.s[i]> b.s[i])?(int16_t)-1:0; return r; }

// compare epi32
static inline __m128i _mm_cmpeq_epi32(__m128i a, __m128i b)           { __m128i r{}; for(int i=0;i<4;i++) r.i[i]=(a.i[i]==b.i[i])?-1:0; return r; }
static inline __m128i _mm_cmpgt_epi32(__m128i a, __m128i b)           { __m128i r{}; for(int i=0;i<4;i++) r.i[i]=(a.i[i]> b.i[i])?-1:0; return r; }

// add/sub epi16/epi32
static inline __m128i _mm_add_epi16(__m128i a, __m128i b)             { __m128i r{}; for(int i=0;i<8;i++) r.s[i]=(int16_t)(a.s[i]+b.s[i]); return r; }
static inline __m128i _mm_sub_epi16(__m128i a, __m128i b)             { __m128i r{}; for(int i=0;i<8;i++) r.s[i]=(int16_t)(a.s[i]-b.s[i]); return r; }
static inline __m128i _mm_add_epi32(__m128i a, __m128i b)             { __m128i r{}; for(int i=0;i<4;i++) r.i[i]=a.i[i]+b.i[i]; return r; }
static inline __m128i _mm_sub_epi32(__m128i a, __m128i b)             { __m128i r{}; for(int i=0;i<4;i++) r.i[i]=a.i[i]-b.i[i]; return r; }
static inline __m128i _mm_add_epi64(__m128i a, __m128i b)             { __m128i r{}; for(int i=0;i<2;i++) r.l[i]=a.l[i]+b.l[i]; return r; }

// shift epi16
static inline __m128i _mm_slli_epi16(__m128i a, int n)                { __m128i r{}; for(int i=0;i<8;i++) r.s[i]=(int16_t)((uint16_t)a.s[i]<<n); return r; }
static inline __m128i _mm_srli_epi16(__m128i a, int n)                { __m128i r{}; for(int i=0;i<8;i++) r.s[i]=(int16_t)((uint16_t)a.s[i]>>n); return r; }
static inline __m128i _mm_srai_epi16(__m128i a, int n)                { __m128i r{}; for(int i=0;i<8;i++) r.s[i]=(int16_t)(a.s[i]>>n); return r; }
// shift epi32
static inline __m128i _mm_slli_epi32(__m128i a, int n)                { __m128i r{}; for(int i=0;i<4;i++) r.ui[i]=(uint32_t)a.ui[i]<<n; return r; }
static inline __m128i _mm_srli_epi32(__m128i a, int n)                { __m128i r{}; for(int i=0;i<4;i++) r.ui[i]=(uint32_t)a.ui[i]>>n; return r; }
static inline __m128i _mm_srai_epi32(__m128i a, int n)                { __m128i r{}; for(int i=0;i<4;i++) r.i[i]=a.i[i]>>n; return r; }

// shuffle / permute / unpack
static inline __m128i _mm_shuffle_epi8(__m128i a, __m128i mask)       { __m128i r{}; for(int i=0;i<16;i++){ int idx=mask.c[i]&0xf; r.c[i]=(mask.c[i]&0x80)?0:a.c[idx]; } return r; }
static inline __m128i _mm_shuffle_epi32(__m128i a, int imm)           { __m128i r{}; for(int i=0;i<4;i++) r.i[i]=a.i[(imm>>(2*i))&3]; return r; }
static inline __m128i _mm_shufflelo_epi16(__m128i a, int imm)         { __m128i r=a; for(int i=0;i<4;i++) r.s[i]=a.s[(imm>>(2*i))&3]; return r; }
static inline __m128i _mm_shufflehi_epi16(__m128i a, int imm)         { __m128i r=a; for(int i=0;i<4;i++) r.s[4+i]=a.s[4+((imm>>(2*i))&3)]; return r; }
static inline __m128i _mm_unpacklo_epi8 (__m128i a, __m128i b)        { __m128i r{}; for(int i=0;i<8;i++){r.c[2*i]=a.c[i];r.c[2*i+1]=b.c[i];} return r; }
static inline __m128i _mm_unpackhi_epi8 (__m128i a, __m128i b)        { __m128i r{}; for(int i=0;i<8;i++){r.c[2*i]=a.c[8+i];r.c[2*i+1]=b.c[8+i];} return r; }
static inline __m128i _mm_unpacklo_epi16(__m128i a, __m128i b)        { __m128i r{}; for(int i=0;i<4;i++){r.s[2*i]=a.s[i];r.s[2*i+1]=b.s[i];} return r; }
static inline __m128i _mm_unpackhi_epi16(__m128i a, __m128i b)        { __m128i r{}; for(int i=0;i<4;i++){r.s[2*i]=a.s[4+i];r.s[2*i+1]=b.s[4+i];} return r; }
static inline __m128i _mm_unpacklo_epi32(__m128i a, __m128i b)        { __m128i r{}; r.i[0]=a.i[0];r.i[1]=b.i[0];r.i[2]=a.i[1];r.i[3]=b.i[1]; return r; }
static inline __m128i _mm_unpackhi_epi32(__m128i a, __m128i b)        { __m128i r{}; r.i[0]=a.i[2];r.i[1]=b.i[2];r.i[2]=a.i[3];r.i[3]=b.i[3]; return r; }

// pack
static inline __m128i _mm_packs_epi32(__m128i a, __m128i b) {
    __m128i r{};
    for(int i=0;i<4;i++) r.s[i]  =(int16_t)(a.i[i]<-32768?-32768:a.i[i]>32767?32767:a.i[i]);
    for(int i=0;i<4;i++) r.s[4+i]=(int16_t)(b.i[i]<-32768?-32768:b.i[i]>32767?32767:b.i[i]);
    return r;
}
static inline __m128i _mm_packus_epi16(__m128i a, __m128i b) {
    __m128i r{};
    for(int i=0;i<8;i++) r.uc[i]  =(uint8_t)(a.s[i]<0?0:a.s[i]>255?255:a.s[i]);
    for(int i=0;i<8;i++) r.uc[8+i]=(uint8_t)(b.s[i]<0?0:b.s[i]>255?255:b.s[i]);
    return r;
}

// movemask / popcnt
static inline int          _mm_movemask_epi8(__m128i a)               { int r=0; for(int i=0;i<16;i++) r|=((a.uc[i]>>7)&1)<<i; return r; }
static inline unsigned int _mm_popcnt_u32(unsigned int v)             { unsigned int c=0; while(v){c+=v&1;v>>=1;} return c; }
static inline unsigned int _mm_popcnt_u64(uint64_t v)                 { unsigned int c=0; while(v){c+=v&1;v>>=1;} return c; }

// ── __m256 float ops ──────────────────────────────────────────────────────────
static inline __m256  _mm256_setzero_ps()                              { return __m256{}; }
static inline __m256  _mm256_set1_ps(float v)                          { __m256 r{}; for(int i=0;i<8;i++) r.f[i]=v; return r; }
static inline __m256  _mm256_loadu_ps(const float* p)                  { __m256 r{}; memcpy(&r,p,32); return r; }
static inline void    _mm256_storeu_ps(float* p, __m256 a)             { memcpy(p,&a,32); }
static inline __m256  _mm256_add_ps(__m256 a, __m256 b)               { __m256 r{}; for(int i=0;i<8;i++) r.f[i]=a.f[i]+b.f[i]; return r; }
static inline __m256  _mm256_sub_ps(__m256 a, __m256 b)               { __m256 r{}; for(int i=0;i<8;i++) r.f[i]=a.f[i]-b.f[i]; return r; }
static inline __m256  _mm256_mul_ps(__m256 a, __m256 b)               { __m256 r{}; for(int i=0;i<8;i++) r.f[i]=a.f[i]*b.f[i]; return r; }
static inline __m256  _mm256_div_ps(__m256 a, __m256 b)               { __m256 r{}; for(int i=0;i<8;i++) r.f[i]=a.f[i]/b.f[i]; return r; }
static inline __m256  _mm256_min_ps(__m256 a, __m256 b)               { __m256 r{}; for(int i=0;i<8;i++) r.f[i]=a.f[i]<b.f[i]?a.f[i]:b.f[i]; return r; }
static inline __m256  _mm256_max_ps(__m256 a, __m256 b)               { __m256 r{}; for(int i=0;i<8;i++) r.f[i]=a.f[i]>b.f[i]?a.f[i]:b.f[i]; return r; }
static inline __m256  _mm256_and_ps(__m256 a, __m256 b)               { __m256 r{}; uint32_t ai,bi,ri; for(int i=0;i<8;i++){memcpy(&ai,&a.f[i],4);memcpy(&bi,&b.f[i],4);ri=ai&bi;memcpy(&r.f[i],&ri,4);} return r; }
static inline __m256  _mm256_or_ps (__m256 a, __m256 b)               { __m256 r{}; uint32_t ai,bi,ri; for(int i=0;i<8;i++){memcpy(&ai,&a.f[i],4);memcpy(&bi,&b.f[i],4);ri=ai|bi;memcpy(&r.f[i],&ri,4);} return r; }
static inline __m256  _mm256_andnot_ps(__m256 a, __m256 b)            { __m256 r{}; uint32_t ai,bi,ri; for(int i=0;i<8;i++){memcpy(&ai,&a.f[i],4);memcpy(&bi,&b.f[i],4);ri=(~ai)&bi;memcpy(&r.f[i],&ri,4);} return r; }
static inline __m256  _mm256_cmp_ps(__m256 a, __m256 b, int imm) {
    __m256 r{}; uint32_t mask;
    for(int i=0;i<8;i++){
        bool res=false;
        switch(imm&0x1f){ case 0:res=(a.f[i]==b.f[i]); break; case 1:res=(a.f[i]<b.f[i]); break;
                          case 2:res=(a.f[i]<=b.f[i]); break; case 4:res=(a.f[i]!=b.f[i]); break;
                          case 5:res=!(a.f[i]< b.f[i]); break; case 14:res=(a.f[i]> b.f[i]); break;
                          case 13:res=(a.f[i]>=b.f[i]); break; default:res=false; }
        mask=res?0xffffffff:0; memcpy(&r.f[i],&mask,4);
    } return r;
}
static inline __m256  _mm256_blendv_ps(__m256 a, __m256 b, __m256 mask) { __m256 r{}; uint32_t m; for(int i=0;i<8;i++){memcpy(&m,&mask.f[i],4);r.f[i]=(m>>31)?b.f[i]:a.f[i];} return r; }
static inline int     _mm256_movemask_ps(__m256 a)                    { int r=0; uint32_t m; for(int i=0;i<8;i++){memcpy(&m,&a.f[i],4);r|=((m>>31)&1)<<i;} return r; }
static inline __m256  _mm256_permutevar8x32_ps(__m256 s, __m256i idx) { __m256 r{}; for(int i=0;i<8;i++) r.f[i]=s.f[idx.i[i]&7]; return r; }
static inline __m256i _mm256_castps_si256(__m256 a)                    { __m256i r{}; memcpy(&r,&a,32); return r; }
static inline __m256  _mm256_castsi256_ps(__m256i a)                   { __m256 r{}; memcpy(&r,&a,32); return r; }

// ── __m256i integer ops ───────────────────────────────────────────────────────
static inline __m256i _mm256_setzero_si256()                           { return __m256i{}; }
static inline __m256i _mm256_loadu_si256(const __m256i* p)             { __m256i r{}; memcpy(&r,p,32); return r; }
static inline void    _mm256_storeu_si256(__m256i* p, __m256i a)       { memcpy(p,&a,32); }

// set1
static inline __m256i _mm256_set1_epi8(int8_t v)                       { __m256i r{}; for(int i=0;i<32;i++)  r.c[i]=v; return r; }
static inline __m256i _mm256_set1_epi16(int16_t v)                     { __m256i r{}; for(int i=0;i<16;i++) r.s[i]=v; return r; }
static inline __m256i _mm256_set1_epi32(int32_t v)                     { __m256i r{}; for(int i=0;i<8;i++)  r.i[i]=v; return r; }

// add/sub epi16/epi32
static inline __m256i _mm256_add_epi16(__m256i a, __m256i b)          { __m256i r{}; for(int i=0;i<16;i++) r.s[i]=(int16_t)(a.s[i]+b.s[i]); return r; }
static inline __m256i _mm256_sub_epi16(__m256i a, __m256i b)          { __m256i r{}; for(int i=0;i<16;i++) r.s[i]=(int16_t)(a.s[i]-b.s[i]); return r; }
static inline __m256i _mm256_add_epi32(__m256i a, __m256i b)          { __m256i r{}; for(int i=0;i<8;i++) r.i[i]=a.i[i]+b.i[i]; return r; }
static inline __m256i _mm256_sub_epi32(__m256i a, __m256i b)          { __m256i r{}; for(int i=0;i<8;i++) r.i[i]=a.i[i]-b.i[i]; return r; }

// bitwise
static inline __m256i _mm256_and_si256(__m256i a, __m256i b)          { __m256i r{}; for(int i=0;i<4;i++) r.l[i]=a.l[i]&b.l[i]; return r; }
static inline __m256i _mm256_or_si256 (__m256i a, __m256i b)          { __m256i r{}; for(int i=0;i<4;i++) r.l[i]=a.l[i]|b.l[i]; return r; }
static inline __m256i _mm256_andnot_si256(__m256i a, __m256i b)       { __m256i r{}; for(int i=0;i<4;i++) r.l[i]=(~a.l[i])&b.l[i]; return r; }

// compare
static inline __m256i _mm256_cmpeq_epi32(__m256i a, __m256i b)        { __m256i r{}; for(int i=0;i<8;i++) r.i[i]=(a.i[i]==b.i[i])?-1:0; return r; }
static inline __m256i _mm256_cmpgt_epi32(__m256i a, __m256i b)        { __m256i r{}; for(int i=0;i<8;i++) r.i[i]=(a.i[i]> b.i[i])?-1:0; return r; }

// shift epi32
static inline __m256i _mm256_slli_epi32(__m256i a, int n)             { __m256i r{}; for(int i=0;i<8;i++) r.ui[i]=(uint32_t)a.ui[i]<<n; return r; }
static inline __m256i _mm256_srli_epi32(__m256i a, int n)             { __m256i r{}; for(int i=0;i<8;i++) r.ui[i]=(uint32_t)a.ui[i]>>n; return r; }

// shuffle / permute
static inline __m256i _mm256_permutevar8x32_epi32(__m256i a, __m256i idx) { __m256i r{}; for(int i=0;i<8;i++) r.i[i]=a.i[idx.i[i]&7]; return r; }

// cvt
static inline __m256i _mm256_cvtepu8_epi32(__m128i a)                 { __m256i r{}; for(int i=0;i<8;i++) r.i[i]=(int32_t)(uint8_t)a.c[i]; return r; }

// macros (must be after the inline functions)
#define _mm256_setr_ps(a,b,c,d,e,f,g,h)  (__m256{{a,b,c,d,e,f,g,h}})
static inline __m256i _mm256_setr_epi8(
    int8_t a0, int8_t a1, int8_t a2,  int8_t a3,  int8_t a4,  int8_t a5,  int8_t a6,  int8_t a7,
    int8_t a8, int8_t a9, int8_t a10, int8_t a11, int8_t a12, int8_t a13, int8_t a14, int8_t a15,
    int8_t b0, int8_t b1, int8_t b2,  int8_t b3,  int8_t b4,  int8_t b5,  int8_t b6,  int8_t b7,
    int8_t b8, int8_t b9, int8_t b10, int8_t b11, int8_t b12, int8_t b13, int8_t b14, int8_t b15)
{
    __m256i r{};
    r.c[0]=a0;  r.c[1]=a1;  r.c[2]=a2;  r.c[3]=a3;
    r.c[4]=a4;  r.c[5]=a5;  r.c[6]=a6;  r.c[7]=a7;
    r.c[8]=a8;  r.c[9]=a9;  r.c[10]=a10;r.c[11]=a11;
    r.c[12]=a12;r.c[13]=a13;r.c[14]=a14;r.c[15]=a15;
    r.c[16]=b0; r.c[17]=b1; r.c[18]=b2; r.c[19]=b3;
    r.c[20]=b4; r.c[21]=b5; r.c[22]=b6; r.c[23]=b7;
    r.c[24]=b8; r.c[25]=b9; r.c[26]=b10;r.c[27]=b11;
    r.c[28]=b12;r.c[29]=b13;r.c[30]=b14;r.c[31]=b15;
    return r;
}

#include <cmath>

// ── Apple Silicon extras ──────────────────────────────────────────────────────
static inline __m128i _mm_insert_epi16(__m128i a, int v, int idx)
    { __m128i r=a; r.s[idx&7]=(int16_t)v; return r; }

static inline __m256i _mm256_setr_epi32(int a,int b,int c,int d,int e,int f,int g,int h)
    { __m256i r{}; r.i[0]=a;r.i[1]=b;r.i[2]=c;r.i[3]=d;r.i[4]=e;r.i[5]=f;r.i[6]=g;r.i[7]=h; return r; }

#ifndef _CMP_LE_OS
#define _CMP_EQ_OQ    0
#define _CMP_LT_OS    1
#define _CMP_LE_OS    2
#define _CMP_UNORD_Q  3
#define _CMP_NEQ_UQ   4
#define _CMP_NLT_US   5
#define _CMP_NLE_US   6
#define _CMP_ORD_Q    7
#define _CMP_GE_OS    13
#define _CMP_GT_OS    14
#endif

#endif  // end of ARM scalar stand-ins

#ifdef __linux__
#include <malloc.h>
#endif

/*
   IMPORTANT: These are not general purpose data structures. They have been
   designed exclusively for TreeNSearch and might be unsafe if used in
   other applications.
*/

namespace tns
{
   namespace internals
   {
      /*
         Aligned allocator so that vectorized types can be used in std containers
         from: https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
      */
      template <typename T, std::size_t N = 32>
      class AlignmentAllocator {
      public:
         typedef T value_type;
         typedef std::size_t size_type;
         typedef std::ptrdiff_t difference_type;

         typedef T* pointer;
         typedef const T* const_pointer;

         typedef T& reference;
         typedef const T& const_reference;

      public:
         inline AlignmentAllocator() throw () { }

         template <typename T2>
         inline AlignmentAllocator(const AlignmentAllocator<T2, N>&) throw () { }

         inline ~AlignmentAllocator() throw () { }

         inline pointer address(reference r)             { return &r; }
         inline const_pointer address(const_reference r) const { return &r; }

         inline pointer allocate(size_type n) {
#ifdef _WIN32
            return (pointer)_aligned_malloc(n * sizeof(value_type), N);
#else
            void* ptr = nullptr;
            if (posix_memalign(&ptr, N, n * sizeof(value_type)) != 0)
               throw std::bad_alloc();
            return (pointer)ptr;
#endif
         }

         inline void deallocate(pointer p, size_type) {
#ifdef _WIN32
            _aligned_free(p);
#else
            free(p);
#endif
         }

         inline void construct(pointer p, const value_type& wert) {
            new (p) value_type(wert);
         }

         inline void destroy(pointer p) {
            p->~value_type();
         }

         inline size_type max_size() const throw () {
            return size_type(-1) / sizeof(value_type);
         }

         template <typename T2>
         struct rebind {
            typedef AlignmentAllocator<T2, N> other;
         };

         bool operator!=(const AlignmentAllocator<T, N>& other) const {
            return !(*this == other);
         }

         bool operator==(const AlignmentAllocator<T, N>& other) const {
            return true;
         }
      };

      /**
      * Alias for aligned vector
      */
      template<typename T, size_t SIZE>
      using avector = std::vector<T, AlignmentAllocator<T, SIZE>>;

      /**
      * Collection of fixed sized dynamically allocated memory chunks, used
      * to store the neighbor lists.
      */
      template<typename T, size_t CHUNKSIZE = 1000>
      class chunked_vector
      {
      private:
         std::vector<std::array<T*, 2>> chunks; // [{begin, cursor}]
         int current = 0;

      public:
         chunked_vector(const chunked_vector<T, CHUNKSIZE>& other) = delete;
         chunked_vector(chunked_vector<T, CHUNKSIZE>&& other) noexcept = default;
         chunked_vector()
         {
            T* ptr = new T[CHUNKSIZE];
            this->chunks.push_back({ ptr, ptr });
            this->current = 0;
         };
         ~chunked_vector()
         {
            for (std::array<T*, 2>& chunk : this->chunks)
               delete[] chunk[0];
         }
         size_t get_chunk_size() { return CHUNKSIZE; }
         T* get_cursor_with_space_to_write(const size_t n)
         {
            if (n > CHUNKSIZE) {
               std::cout << "chunked_vector: Cannot allow_to_append n > CHUNKSIZE ("
                         << n << " < " << CHUNKSIZE << ")." << std::endl;
               exit(-1);
            }
            const size_t space_left = CHUNKSIZE -
               (size_t)std::distance(this->chunks[this->current][0], this->chunks[this->current][1]) - 1;
            if (n > space_left) {
               this->current++;
               if ((size_t)this->chunks.size() == (size_t)this->current) {
                  T* ptr = new T[CHUNKSIZE];
                  this->chunks.push_back({ ptr, ptr });
               }
               this->chunks[this->current][1] = this->chunks[this->current][0];
            }
            T* return_cursor = this->chunks[this->current][1];
            this->chunks[this->current][1] += n;
            *this->chunks[this->current][1] = -1;
            return return_cursor;
         }
         void clear()
         {
            this->chunks[0][1] = this->chunks[0][0];
            this->current = 0;
         }
         size_t n_bytes() const
         {
            return this->chunks.size() * CHUNKSIZE * static_cast<size_t>(sizeof(T));
         }
      };

      /**
      * Uninitialized vector. Allows resize without initializing newly allocated memory.
      */
      template<typename T>
      class uvector
      {
      public:
         T* data   = nullptr;
         T* end    = nullptr;
         T* cursor = nullptr;

         uvector() = default;
         ~uvector()
         {
            if (this->data != nullptr)
               delete[] this->data;
         }
         uvector(const uvector& other)
         {
            if (other.data != nullptr) {
               this->data   = new T[std::distance(other.data, other.end)];
               this->end    = this->data + std::distance(other.data, other.end);
               this->cursor = this->data + std::distance(other.data, other.cursor);
               memcpy(this->data, other.data, sizeof(T) * std::distance(other.data, other.end));
            }
         }
         uvector& operator=(uvector other)
         {
            std::swap(this->data,   other.data);
            std::swap(this->end,    other.end);
            std::swap(this->cursor, other.cursor);
            return *this;
         }
         void init(const int n)
         {
            assert(n >= 0);
            if (this->data != nullptr) {
               delete[] this->data;
               this->data = this->end = this->cursor = nullptr;
            }
            if (n > 0) {
               this->data   = new T[n];
               this->cursor = this->data;
               this->end    = this->data + n;
            }
         }
         void init_with_at_least_size(const int n, const double multiplier = 1.0)
         {
            assert(n >= 0);
            if (this->capacity() < n)
               this->init((int)(multiplier * n));
            else
               this->cursor = this->data;
         }
         void grow_while_keeping_data(const int n)
         {
            assert(n >= 0);
            if (n > this->capacity()) {
               size_t cursor_ = std::distance(this->data, this->cursor);
               T* data_ = new T[n];
               memcpy(data_, this->data, sizeof(T) * this->capacity());
               delete[] this->data;
               this->data   = data_;
               this->cursor = this->data + cursor_;
               this->end    = this->data + n;
            }
         }
         int capacity()      const { return (int)std::distance(this->data,   this->end);    }
         int capacity_left() const { return (int)std::distance(this->cursor, this->end);    }
         int size()          const { return (int)std::distance(this->data,   this->cursor); }

         template<typename INDEX_TYPE> T& operator[](INDEX_TYPE idx)       { return this->data[idx]; }
         template<typename INDEX_TYPE> T  operator[](INDEX_TYPE idx) const { return this->data[idx]; }
      };
   }
}
