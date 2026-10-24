#ifndef DILITHIUM_ROUNDING_H
#define DILITHIUM_ROUNDING_H

#include <stdint.h>

uint32_t dilithium_power2round(const uint32_t a, uint32_t *a0);
uint32_t dilithium_decompose(uint32_t a, uint32_t *a0);
unsigned int dilithium_make_hint(const uint32_t a0, const uint32_t a1);
uint32_t dilithium_use_hint(const uint32_t a, const unsigned int hint);

#endif