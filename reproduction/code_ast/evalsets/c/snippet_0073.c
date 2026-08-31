/* types.h - v0.90 - MIT - <NAME>, 2019 
 * 
 * LICENSE
 *   See end of file for license information.
 */

#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include <stdint.h>

typedef char       i8;
typedef int16_t   i16;
typedef int32_t   i32;
typedef int64_t   i64;

typedef uint8_t   u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef u64      umm;
typedef i64      imm;

typedef float    f32;
typedef double   f64;

typedef i32      bool;
typedef i32      b32;
typedef i8       b8;
typedef i64      b64;

#define true  (1 == 1)
#define false (1 == 0)

#define  U8_MAX 0xFF
#define U16_MAX 0xFFFF
#define U32_MAX 0xFFFFFFFF
#define U64_MAX 0xFFFFFFFFFFFFFFFF

#define I8_MAX  0x7F
#define I16_MAX 0x7FFF
#define I32_MAX 0x7FFFFFFF
#define I64_MAX 0x7FFFFFFFFFFFFFFF

#define I8_MIN  0x80
#define I16_MIN 0x8000
#define I32_MIN 0x80000000
#define I64_MIN 0x8000000000000000

#define array_len(a) (sizeof((a))/sizeof((a)[0]))

#define offsetof(struct_name, member_name) \
	((umm)&(((struct_name*)0)->member_name))


#if defined(__clang__)
#define BNI_HAVE_VA_ARGS
typedef __builtin_va_list va_list;
#define va_start(ap, param) __builtin_va_start(ap, param)
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)
#endif

#define global static
#define internal static
#define local_assign_once static

#endif // TYPES_H_INCLUDED

/* Copyright 2019 <NAME> <<EMAIL>> 
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
 // vim: ts=4 sw=4 cindent cino=0(,\:0,l1