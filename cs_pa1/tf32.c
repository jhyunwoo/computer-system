#include <stdio.h>
#include "tf32.h"

/*
    Do NOT include any C libraries or header files
    except those above

    use the defines in tf32.h
    if neccessary, you can add some macros below
*/

/* ---------------- TF32 format (19-bit) ----------------
 * [18]      sign
 * [17:10]   exponent (8-bit, bias = 127)
 * [9:0]     fraction (10-bit)
 */

#ifndef TF32_LOCAL_DEFS
#define TF32_LOCAL_DEFS

#define TF32_BITS            19u
#define TF32_SIGN_MASK       0x40000u
#define TF32_EXP_MASK        0x3FC00u
#define TF32_FRAC_MASK       0x003FFu
#define TF32_EXP_SHIFT       10u
#define TF32_EXP_BIAS        127
#define TF32_EXP_INF_NAN     0xFFu

#define U32(x) ((unsigned)(x))

#endif

/* ------------- helpers (no float/double arithmetic) ------------- */

static inline unsigned tf32_sign(tf32 x){ return (x & TF32_SIGN_MASK) != 0u; }
static inline unsigned tf32_exp (tf32 x){ return (x & TF32_EXP_MASK) >> TF32_EXP_SHIFT; }
static inline unsigned tf32_frac(tf32 x){ return (x & TF32_FRAC_MASK); }

static inline int is_nan(tf32 x){ return (tf32_exp(x)==TF32_EXP_INF_NAN) && (tf32_frac(x)!=0u); }
static inline int is_inf(tf32 x){ return (tf32_exp(x)==TF32_EXP_INF_NAN) && (tf32_frac(x)==0u); }
static inline int is_zero(tf32 x){ return ( (x & (~TF32_SIGN_MASK)) == 0u ); }
static inline int is_subnormal(tf32 x){ return (tf32_exp(x)==0u) && (tf32_frac(x)!=0u); }

/* count leading zeros for 32-bit unsigned */
static inline unsigned clz32(unsigned v){
    if(v==0u) return 32u;
    unsigned n=0u;
    if((v & 0xFFFF0000u)==0u){ n+=16u; v<<=16; }
    if((v & 0xFF000000u)==0u){ n+=8u;  v<<=8;  }
    if((v & 0xF0000000u)==0u){ n+=4u;  v<<=4;  }
    if((v & 0xC0000000u)==0u){ n+=2u;  v<<=2;  }
    if((v & 0x80000000u)==0u){ n+=1u; }
    return n;
}

/* index of highest set bit (0..31). v != 0 */
static inline unsigned ilog2_32(unsigned v){
    return 31u - clz32(v);
}

/* Right shift with sticky: returns (v >> sh), but if any discarded bit was 1, the LSB of the result is forced to 1 */
static inline unsigned rshift_sticky_u32(unsigned v, unsigned sh){
    if(sh==0u) return v;
    if(sh>=32u) return (v?1u:0u);
    unsigned dropped_mask = (1u<<sh) - 1u;
    unsigned dropped = v & dropped_mask;
    unsigned res = v >> sh;
    if(dropped) res |= 1u;
    return res;
}

/* Round-to-even on kept value when we had a right shift by k bits.
   kept = v >> k, guard = bit (k-1), sticky = any below (k-1).
   Returns rounded kept. */
static inline unsigned round_to_even_after_rshift(unsigned v, unsigned k){
    if(k==0u) return v;
    if(k>=32u){
        /* everything shifted out: result is 0 or 1 depending on stickiness; tie can't happen here */
        return (v?1u:0u);
    }
    unsigned kept = v >> k;
    unsigned guard = (v >> (k-1u)) & 1u;
    unsigned lower_mask = (1u<<(k-1u)) - 1u;
    unsigned sticky = (v & lower_mask) ? 1u : 0u;
    unsigned lsb = kept & 1u;
    if(guard && (sticky || lsb)){
        kept += 1u;
    }
    return kept;
}

/* Pack sign/exponent/fraction into tf32 (assumes exp and frac properly bounded) */
static inline tf32 pack_tf32(unsigned s, unsigned e, unsigned f){
    return (tf32)((s?TF32_SIGN_MASK:0u) | ((e & 0xFFu) << TF32_EXP_SHIFT) | (f & TF32_FRAC_MASK));
}

/* Make a canonical quiet NaN with sign s */
static inline tf32 tf32_nan(unsigned s){
    return pack_tf32(s, TF32_EXP_INF_NAN, 1u);
}

/* Saturated infinities */
static inline tf32 tf32_pinf(void){ return pack_tf32(0u, TF32_EXP_INF_NAN, 0u); }
static inline tf32 tf32_ninf(void){ return pack_tf32(1u, TF32_EXP_INF_NAN, 0u); }

/* Normalize (mant with implicit 1 for normal case) -> produce (e,f10) with round-to-even.
   Input:
     mant:  unsigned, at least 11+extra bits precision (we'll provide proper width in callers)
     e    : unbiased exponent corresponding to mant being in fixed scale where the hidden 1 should end up at bit 10.
     We assume mant has top bit >= (1<<10) (or we will shift left and decrement e).
   Output: packed tf32 or subnormal/overflow as needed. */
static tf32 normalize_pack(unsigned sign, int e_unbiased, unsigned mant /* >= 11 bits */){
    /* Ensure mant has leading 1 at bit 10 (i.e., mant in [1<<10, 2<<10) ) by shifting */
    if(mant==0u){
        /* zero */
        return pack_tf32(sign, 0u, 0u);
    }

    /* Find actual top bit */
    unsigned msb = ilog2_32(mant);
    /* We want msb == 10 for normalized (so that 1.xxx with 10 fraction bits). */
    if(msb > 10u){
        unsigned k = msb - 10u;
        /* Round when shifting right by k */
        unsigned rounded = round_to_even_after_rshift(mant, k);
        mant = rounded;
        e_unbiased += (int)k;
        /* Rounding could cause carry into bit 11 */
        if(mant >= (1u<<11)){
            mant >>= 1;
            e_unbiased += 1;
        }
    }else if(msb < 10u){
        unsigned k = 10u - msb;
        if(e_unbiased - (int)k >= -126){ /* we can still keep as normal after left-shift */
            mant <<= k;
            e_unbiased -= (int)k;
        }else{
            /* Will become subnormal; handle after this block */
        }
    }

    /* Now try to form normal number if exponent in range */
    if(e_unbiased + TF32_EXP_BIAS >= 1 && e_unbiased + TF32_EXP_BIAS <= 254){
        unsigned exp_field = (unsigned)(e_unbiased + TF32_EXP_BIAS);
        unsigned frac10 = mant & ((1u<<10)-1u);
        return pack_tf32(sign, exp_field, frac10);
    }

    /* Overflow to infinity */
    if(e_unbiased + TF32_EXP_BIAS >= 255){
        return sign ? tf32_ninf() : tf32_pinf();
    }

    /* Underflow -> subnormal or zero.
       For subnormal, exponent field = 0, value = mant * 2^(e_unbiased - 10).
       We must shift right by s = (1 - bias) - e_unbiased to move exponent to 1-bias,
       but since mant is scaled with hidden at bit10, total shift to get 10-bit fraction is:
       shift = (1 - TF32_EXP_BIAS) - e_unbiased  +  (??)
       More directly: target fraction f = round_to_even( mant * 2^(e_unbiased - (1 - bias)) ), but mant is scaled by 2^10.
       Simpler: We have current mant with implicit '1' at bit10 (or lower if earlier). To create subnormal, we need to
       right shift so that exponent becomes 1-bias and hidden bit disappears, keeping 10 fraction bits.
    */
    {
        /* Compute how many positions to shift to produce 10-bit fraction at exponent=0 */
        int distance = (1 - TF32_EXP_BIAS) - e_unbiased; /* how far below min normal exponent */
        /* Current mant has 11 significant bits; to get only 10 fraction bits (without hidden 1), add +1 */
        int total_shift = distance + 1; /* remove hidden bit */
        if(total_shift <= 0){
            /* shouldn't generally happen here; treat as normal edge */
            unsigned exp_field = 1u;
            unsigned frac10 = (mant & ((1u<<10)-1u));
            return pack_tf32(sign, exp_field, frac10);
        }
        unsigned u_total_shift = (unsigned)total_shift;
        unsigned f = round_to_even_after_rshift(mant, u_total_shift);
        if(f >= (1u<<10)){
            /* Rounded up overflows into min normal */
            return pack_tf32(sign, 1u, 0u);
        }
        return pack_tf32(sign, 0u, f);
    }
}

/* Decode tf32 into sign, unbiased exponent, mant (with hidden 1 for normals, plain for subnormals), and a flag normal/subnormal/zero */
static void decode_tf32(tf32 x, unsigned* ps, int* pe_unb, unsigned* pmant, int* p_is_zero, int* p_is_inf, int* p_is_nan, int* p_is_sub){
    unsigned s = tf32_sign(x);
    unsigned e = tf32_exp(x);
    unsigned f = tf32_frac(x);
    *ps = s;
    *p_is_nan = (e==TF32_EXP_INF_NAN && f!=0u);
    *p_is_inf = (e==TF32_EXP_INF_NAN && f==0u);
    *p_is_zero = (e==0u && f==0u);
    *p_is_sub  = (e==0u && f!=0u);
    if(e==0u){
        if(f==0u){
            *pe_unb = 0;
            *pmant = 0u;
        }else{
            *pe_unb = 1 - TF32_EXP_BIAS; /* subnormal exponent */
            *pmant  = f;                 /* no hidden 1 */
        }
    }else if(e==TF32_EXP_INF_NAN){
        *pe_unb = 0;
        *pmant  = 0u;
    }else{
        *pe_unb = (int)e - TF32_EXP_BIAS;
        *pmant  = (1u<<10) | f; /* hidden 1 */
    }
}

/* Align two mantissas to the same exponent with sticky management.
   Inputs: mant a (ma), exponent a (Ea), mant b (mb), exponent b (Eb).
   Output: exponent E = max(Ea,Eb), and adjusted ma', mb' (with sticky in LSB after shifts).
   We store extra headroom bits; use 3 extra bits for rounding later: we shift mant<<3 before alignment.
*/
static void align_mantissas(unsigned* pma, int* pEa, unsigned* pmb, int* pEb, int* pE_out){
    unsigned ma = *pma << 3; /* add 3 zero bits for GRS positions */
    unsigned mb = *pmb << 3;
    int Ea = *pEa;
    int Eb = *pEb;
    int E = (Ea > Eb)? Ea : Eb;
    unsigned shift_a = (unsigned)(E - Ea);
    unsigned shift_b = (unsigned)(E - Eb);
    ma = rshift_sticky_u32(ma, shift_a);
    mb = rshift_sticky_u32(mb, shift_b);
    *pma = ma;
    *pmb = mb;
    *pE_out = E;
}

/* Final rounding from extended mantissa (bits [13..3] kept, bit2 guard, bit1 unused, bit0 sticky).
   Returns kept 11-bit (1 hidden + 10 frac), updates exponent on overflow.
*/
static unsigned round_from_ext(unsigned* pMantExt, int* pE){
    unsigned m = *pMantExt;
    /* kept bits: [13..3] (11 bits) */
    unsigned kept = (m >> 3) & 0x7FFu;
    unsigned guard = (m >> 2) & 1u;
    unsigned sticky = (m & 0x3u) ? 1u : 0u; /* include bit1 + bit0 as 'sticky-any' */
    unsigned lsb = kept & 1u;
    if(guard && (sticky || lsb)){
        kept += 1u;
        if(kept >= 0x800u){ /* 1<<11 */
            kept >>= 1;
            *pE += 1;
        }
    }
    return kept;
}

/* ---------------- Conversions ---------------- */

tf32 int2tf32(int in) {
    if(in==0) return pack_tf32(0u, 0u, 0u);
    unsigned s = (in < 0) ? 1u : 0u;

    /* absolute value as unsigned (handle INT_MIN safely) */
    unsigned u = s ? (unsigned)(-(long long)in) : (unsigned)in;

    /* find msb */
    unsigned msb = ilog2_32(u); /* 0..31 */
    int e_unb = (int)msb;       /* since value = u = 1<<msb * (1 + rem/2^msb) */

    /* build mantissa with 11 kept bits (1 hidden + 10 frac), with round-to-even */
    if(msb <= 10u){
        unsigned mant = u << (10u - msb); /* exact, no rounding */
        return normalize_pack(s, e_unb - 10, mant);
    }else{
        unsigned shift = msb - 10u;
        unsigned mant = round_to_even_after_rshift(u, shift);
        /* Possible carry */
        if(mant >= (1u<<11)){
            mant >>= 1;
            e_unb += 1;
        }
        return normalize_pack(s, e_unb - 10, mant);
    }
}

int tf322int(tf32 in) {
    unsigned s = tf32_sign(in);
    unsigned e = tf32_exp(in);
    unsigned f = tf32_frac(in);

    if(e==TF32_EXP_INF_NAN){
        if(f!=0u){
            /* NaN -> TMin per table */
            return 0x80000000; /* INT_MIN */
        }else{
            return s ? 0x80000000 : 0x7FFFFFFF;
        }
    }
    if(e==0u){
        /* zero or subnormal -> |value| < 1, round-to-even -> 0 (sign ignored) */
        return 0;
    }

    /* normal */
    int e_unb = (int)e - TF32_EXP_BIAS;
    unsigned mant = (1u<<10) | f; /* 11 bits fixed point (scaled by 2^10) */
    /* value = mant * 2^(e_unb - 10) */

    if(e_unb >= 10){
        /* left shift: check overflow using 64-bit */
        unsigned long long v = ((unsigned long long)mant) << (unsigned)(e_unb - 10);
        if(s==0u){
            if(v > 0x7FFFFFFFULL) return 0x7FFFFFFF;
            return (int)v;
        }else{
            if(v > 0x80000000ULL) return 0x80000000;
            return -(int)v;
        }
    }else{
        /* right shift with round-to-even */
        unsigned k = (unsigned)(10 - e_unb);
        unsigned rounded = round_to_even_after_rshift(mant, k);
        if(s==0u){
            if(rounded > 0x7FFFFFFFu) return 0x7FFFFFFF;
            return (int)rounded;
        }else{
            if(rounded > 0x80000000u) return 0x80000000;
            return -(int)rounded;
        }
    }
}

tf32 double2tf32(double in) {
    union { double f; unsigned long long u; } x;
    x.f = in;
    unsigned long long ubits = x.u;

    unsigned s = (unsigned)((ubits >> 63) & 1ull);
    unsigned exp11 = (unsigned)((ubits >> 52) & 0x7FFull);
    unsigned long long frac52 = (ubits & 0xFFFFFFFFFFFFFull);

    if(exp11==0x7FFu){
        if(frac52!=0ull) return tf32_nan(s);
        return pack_tf32(s, TF32_EXP_INF_NAN, 0u);
    }
    if(exp11==0u){
        /* zero or subnormal double -> extremely small, maps to ±0 in TF32 (or tiny subnormals) */
        if(frac52==0ull){
            return pack_tf32(s, 0u, 0u);
        }else{
            /* Treat as subnormal double ~ 0 => goes to TF32 zero (too tiny), keep sign */
            return pack_tf32(s, 0u, 0u);
        }
    }

    /* normal double */
    int e_d_unb = (int)exp11 - 1023; /* unbiased double exponent */

    /* Build 53-bit mantissa with hidden 1 */
    unsigned long long mant53 = (1ull<<52) | frac52; /* 53 bits */

    /* Map exponent to TF32: e_t = e_d + bias_diff (no mant scaling yet) */
    int e_t = e_d_unb + TF32_EXP_BIAS;

    /* Create 11-bit mant (1+10) with rounding: shift 53 -> 11 => shift by 42 */
    unsigned long long kept = mant53 >> 42;
    unsigned long long guard = (mant53 >> 41) & 1ull;
    unsigned long long sticky = (mant53 & ((1ull<<41) - 1ull)) ? 1ull : 0ull;
    unsigned long long lsb = kept & 1ull;
    if(guard && (sticky || lsb)) kept++;

    if(kept >= (1ull<<11)){
        kept >>= 1;
        e_t += 1;
    }

    if(e_t >= 0xFF){
        return pack_tf32(s, TF32_EXP_INF_NAN, 0u); /* overflow -> inf */
    }
    if(e_t <= 0){
        /* Make subnormal: need only 10-bit fraction; total shift from mant53 to 10 is 43,
           plus extra for exponent underflow of (1 - e_t) */
        int extra = 1 - e_t;
        int total = 43 + extra; /* 53->10 is 43 */
        if(total >= 64){
            return pack_tf32(s, 0u, 0u);
        }
        unsigned long long f = mant53 >> (unsigned)total;
        /* tie-to-even */
        unsigned long long gbit = (mant53 >> (unsigned)(total-1)) & 1ull;
        unsigned long long st = (mant53 & (((1ull<<(unsigned)(total-1)))-1ull)) ? 1ull : 0ull;
        if(gbit && (st || (f & 1ull))) f++;

        if(f >= (1ull<<10)){
            /* bumped into min normal */
            return pack_tf32(s, 1u, 0u);
        }
        return pack_tf32(s, 0u, (unsigned)f);
    }

    /* normal */
    return pack_tf32(s, (unsigned)e_t, (unsigned)(kept & ((1ull<<10)-1ull)));
}

double tf322double(tf32 in) {
    /* Build double purely by bit packing (no float arithmetic). */
    unsigned s = tf32_sign(in);
    unsigned e = tf32_exp(in);
    unsigned f = tf32_frac(in);

    union { double f; unsigned long long u; } y;
    unsigned long long sign64 = ((unsigned long long)s) << 63;

    if(e==TF32_EXP_INF_NAN){
        if(f!=0u){
            y.u = sign64 | (0x7FFull<<52) | 1ull; /* qNaN */
            return y.f;
        }else{
            y.u = sign64 | (0x7FFull<<52);
            return y.f;
        }
    }
    if(e==0u){
        if(f==0u){
            y.u = sign64; /* ±0 */
            return y.f;
        }else{
            /* Subnormal TF32 value: M = f, exponent e_tf = 1 - bias - 10 as shift on M
               Value = f * 2^( (1-127) - 10 ) = f * 2^(-136).
               Represent as normal double:
               Let M = f; msb index of M is m.
               Build double mantissa = M << (52 - m), exponent E_d = (1 - 127 - 10) + m + 1023 = m + 886.
            */
            unsigned M = f;
            unsigned m = ilog2_32(M);
            unsigned long long mant = ((unsigned long long)M) << (52u - m); /* includes hidden 1 at bit 52 if M>=1<<0 */
            unsigned E_d = (unsigned)((1 - TF32_EXP_BIAS - 10) + (int)m + 1023);
            unsigned long long frac_d = mant & ((1ull<<52)-1ull);
            y.u = sign64 | (((unsigned long long)E_d) << 52) | frac_d;
            return y.f;
        }
    }else{
        /* Normal TF32:
           M = (1<<10)+f, msb_pos = 10
           Value = M * 2^(e - 127 - 10)
           Double exponent E_d = (e - 127 - 10) + 10 + 1023 = e + 886
           Double mantissa = M << (52 - 10), and drop hidden bit.
        */
        unsigned M = (1u<<10) | f;
        unsigned long long mant = ((unsigned long long)M) << (52u - 10u);
        unsigned E_d = (unsigned)((int)e + 886);
        unsigned long long frac_d = mant & ((1ull<<52)-1ull);
        y.u = sign64 | (((unsigned long long)E_d) << 52) | frac_d;
        return y.f;
    }
}

/* ---------------- Arithmetic ---------------- */

/* Addition */
tf32 tf32_add(tf32 a, tf32 b) {
    /* Special cases */
    if(is_nan(a)) return a;
    if(is_nan(b)) return b;

    int a_inf = is_inf(a), b_inf = is_inf(b);
    if(a_inf && b_inf){
        /* +inf + -inf => NaN; same sign => that inf */
        if(tf32_sign(a) != tf32_sign(b)) return tf32_nan(0u);
        return a;
    }
    if(a_inf) return a;
    if(b_inf) return b;

    /* Zeros: handle via generic path as well */
    unsigned sa,sb; int Ea,Eb; unsigned Ma,Mb; int za,ia,na,suba, zb,ib,nb,subb;
    decode_tf32(a,&sa,&Ea,&Ma,&za,&ia,&na,&suba);
    decode_tf32(b,&sb,&Eb,&Mb,&zb,&ib,&nb,&subb);

    if(Ma==0u && Mb==0u){
        /* +0 + -0 => +0 (tie-to-even on sign: choose +0) */
        return pack_tf32(0u,0u,0u);
    }

    /* Align mantissas */
    int E;
    align_mantissas(&Ma,&Ea,&Mb,&Eb,&E);

    /* Operate */
    unsigned R;
    unsigned signR;
    if(sa == sb){
        /* same sign: add */
        R = Ma + Mb;
        signR = sa;
        /* normalize if carry into bit 14 */
        if(R & (1u<<(10+4))){ /* bit 14 */
            /* shift right by 1 with sticky */
            unsigned sticky = R & 1u;
            R = (R >> 1) | sticky;
            E += 1;
        }
    }else{
        /* different signs: subtract larger - smaller */
        /* Compare aligned magnitudes */
        if(Ma > Mb){ R = Ma - Mb; signR = sa; }
        else if(Mb > Ma){ R = Mb - Ma; signR = sb; }
        else{
            /* exact cancel -> +0 */
            return pack_tf32(0u,0u,0u);
        }

        /* normalize left until leading '1' of kept (bit13) appears or exponent hits subnormal boundary */
        while((R & (1u<<(10+3)))==0u && E > (1 - TF32_EXP_BIAS)){
            R <<= 1;
            E -= 1;
        }
    }

    /* Round & pack */
    unsigned kept = round_from_ext(&R, &E); /* may bump E */
    int e_unb = E;
    if(kept==0u){
        return pack_tf32(0u,0u,0u);
    }
    return normalize_pack(signR, e_unb - 10, kept);
}

/* Multiplication */
tf32 tf32_mul(tf32 a, tf32 b) {
    if(is_nan(a)) return a;
    if(is_nan(b)) return b;

    int a_inf = is_inf(a), b_inf = is_inf(b);
    int a_zero = is_zero(a), b_zero = is_zero(b);
    unsigned sa = tf32_sign(a), sb = tf32_sign(b);
    unsigned s = sa ^ sb;

    /* Special table */
    if(a_inf || b_inf){
        if(a_zero || b_zero) return tf32_nan(0u); /* ±∞ * 0 => NaN */
        return s ? tf32_ninf() : tf32_pinf();
    }
    if(a_zero || b_zero){
        return pack_tf32(s, 0u, 0u); /* ±0 */
    }

    /* Decode normals/subnormals */
    unsigned Sx,Sy; int Ex,Ey; unsigned Mx,My; int zx,ix,nx,subx,  zy,iy,ny,suby;
    decode_tf32(a,&Sx,&Ex,&Mx,&zx,&ix,&nx,&subx);
    decode_tf32(b,&Sy,&Ey,&My,&zy,&iy,&ny,&suby);

    /* Multiply mantissas: (<= 11 bits each) => up to 22 bits */
    unsigned long long prod = (unsigned long long)Mx * (unsigned long long)My; /* up to ~2^22 */
    /* Determine normalization:
       If prod >= 2^(10+11)=2^21 => leading 1 beyond expected -> shift 11
       else shift 10. */
    int E = Ex + Ey; /* unbiased exponents add */
    unsigned kept;
    if(prod >= (1ull<<21)){
        /* shift by 11, round-to-even */
        kept = (unsigned)(prod >> 11);
        unsigned guard = (unsigned)((prod >> 10) & 1ull);
        unsigned sticky = (unsigned)( (prod & ((1ull<<10)-1ull)) ? 1ull : 0ull );
        if(guard && (sticky || (kept & 1u))) kept++;
        E += 1; /* because we had 2.x */
    }else{
        /* shift by 10 */
        kept = (unsigned)(prod >> 10);
        unsigned guard = (unsigned)((prod >> 9) & 1ull);
        unsigned sticky = (unsigned)( (prod & ((1ull<<9)-1ull)) ? 1ull : 0ull );
        if(guard && (sticky || (kept & 1u))) kept++;
    }

    /* Handle possible carry after rounding */
    if(kept >= (1u<<12)){ /* shouldn't happen, but safe */
        kept >>= 1u;
        E += 1;
    }else if(kept >= (1u<<11)){
        /* move to 11 bits (hidden+10) */
        /* nothing to do, normalize_pack will handle */
    }

    return normalize_pack(s, E - 10, kept);
}

/* Reciprocal with Newton-Raphson (5 iterations): y_{n+1} = y_n * (2 - x * y_n)
   We do iterations in TF32 domain using tf32_add/tf32_mul only (no double arithmetic).
   Initial approximation y0 = 2^{-exp(x)} with sign ignored (we use magnitude here).
*/
static tf32 tf32_abs(tf32 x){ return (tf32)(x & ~TF32_SIGN_MASK); }
static tf32 tf32_neg(tf32 x){ return (tf32)(x ^ TF32_SIGN_MASK); }

static tf32 reciprocal_tf32(tf32 x){
    /* Handle special cases for reciprocal */
    if(is_nan(x)) return x;
    if(is_zero(x)) return tf32_nan(0u);
    if(is_inf(x)) return pack_tf32(0u,0u,0u); /* 1/inf = 0 */

    unsigned s = tf32_sign(x);
    tf32 ax = tf32_abs(x);
    unsigned ex = tf32_exp(ax);
    unsigned fx = tf32_frac(ax);

    /* y0 = 2^{-ex_unbiased} */
    int ex_unb = (ex==0u) ? (1 - TF32_EXP_BIAS) : ((int)ex - TF32_EXP_BIAS);
    /* Build y0 as exact power of two in TF32: exponent field = (-ex_unb) + bias, mant = 0 */
    int y0_e_unb = -ex_unb;
    int y0_e_field = y0_e_unb + TF32_EXP_BIAS;
    tf32 y = (y0_e_field<=0) ? pack_tf32(0u,0u,0u) :
             (y0_e_field>=255) ? tf32_pinf() :
             pack_tf32(0u,(unsigned)y0_e_field,0u);

    /* Iterate 5 times: y = y * (2 - x*y) */
    /* Build constant 2.0 in TF32: exponent (1 + bias), mant=0 */
    tf32 two = pack_tf32(0u, (unsigned)(TF32_EXP_BIAS+1), 0u);

    for(int i=0;i<5;i++){
        tf32 xy = tf32_mul(ax, y);
        tf32 term = tf32_add(two, tf32_neg(xy)); /* (2 - x*y) */
        y = tf32_mul(y, term);
    }

    /* restore sign: 1/x keeps sign of x */
    if(s) y = tf32_neg(y);
    return y;
}

tf32 tf32_div(tf32 a, tf32 b) {
    if(is_nan(a)) return a;
    if(is_nan(b)) return b;

    int a_inf = is_inf(a), b_inf = is_inf(b);
    int a_zero = is_zero(a), b_zero = is_zero(b);
    unsigned s = tf32_sign(a) ^ tf32_sign(b);

    /* Special cases based on table */
    if(a_inf && b_inf) return tf32_nan(s);
    if(a_inf && !b_inf) return s ? tf32_ninf() : tf32_pinf();
    if(!a_inf && b_inf) return pack_tf32(s,0u,0u); /* normal/zero divided by inf -> ±0 */

    if(a_zero && b_zero) return tf32_nan(0u);
    if(a_zero && !b_zero) return pack_tf32(s,0u,0u);
    if(!a_zero && b_zero) return s ? tf32_ninf() : tf32_pinf();

    /* Compute a * (1/b) with Newton-Raphson reciprocal */
    tf32 rinv = reciprocal_tf32(b);
    tf32 res  = tf32_mul(a, rinv);
    return res;
}

/* Comparison: return 1 if a>b, 0 if equal, -1 if a<b; NaN => -2 */
int tf32_compare(tf32 a, tf32 b) {
    if(is_nan(a) || is_nan(b)) return -2;

    /* Handle zeros: +0 == -0 */
    if(is_zero(a) && is_zero(b)) return 0;

    /* If signs differ, positive is greater (except zeros handled) */
    unsigned sa = tf32_sign(a), sb = tf32_sign(b);
    if(sa != sb){
        return sa ? -1 : 1;
    }

    /* Same sign: compare bit patterns with sign-aware order */
    unsigned ea = tf32_exp(a), eb = tf32_exp(b);
    unsigned fa = tf32_frac(a), fb = tf32_frac(b);

    if(a == b) return 0;

    if(sa==0u){
        /* positive: larger exponent wins, then fraction */
        if(ea != eb) return (ea > eb) ? 1 : -1;
        return (fa > fb) ? 1 : -1;
    }else{
        /* negative: more negative is smaller -> reverse */
        if(ea != eb) return (ea > eb) ? -1 : 1;
        return (fa > fb) ? -1 : 1;
    }
}