#!/bin/sh
# Builds the three engine variants. Needs Emscripten (emcc).
set -e
cd "$(dirname "$0")"
F="-O3 -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=33554432 -sEXPORTED_RUNTIME_METHODS=HEAPU8 -sEXPORTED_FUNCTIONS=_model_new,_forward,_logits,_logits_some,_model_cap,_model_grow,_tile_choice,_model_set_threads,_crc32,_malloc,_free"
R="-sEXPORTED_FUNCTIONS=_model_new,_forward,_logits,_logits_some,_model_cap,_model_grow,_tile_choice,_model_set_threads,_crc32,_reader_new,_reader_grow,_reader_forward,_reader_logits,_reader_copy_keys,_reader_mix,_malloc,_free"
# multi-threaded: fixed memory (growth is slow with shared memory); needs a cross-origin isolated page
T="-pthread -sPTHREAD_POOL_SIZE=Math.min(navigator.hardwareConcurrency||4,8)-1 -sINITIAL_MEMORY=201326592 -sALLOW_MEMORY_GROWTH=0"
emcc engine.c $F $T -msimd128 -mrelaxed-simd -o engine-relaxed-mt.mjs
emcc engine.c $F $T -msimd128 -o engine-mt.mjs
emcc engine.c $F $T -o engine-scalar-mt.mjs   # threads without SIMD (Safari 15.2-16.3)
emcc engine.c $F -msimd128 -mrelaxed-simd -o engine-relaxed.mjs   # fused multiply-add (Chrome/Edge 114+)
emcc engine.c $F -msimd128 -o engine.mjs                          # WebAssembly SIMD (2021+ browsers)
emcc engine.c $F -o engine-scalar.mjs                             # any browser with WebAssembly
emcc reader_engine.c $F $R -msimd128 -o reader-simd.mjs           # exact fp32 v10 reader; single worker for low memory
emcc reader_engine.c $F $R -o reader-scalar.mjs                   # fallback when WebAssembly SIMD is unavailable
ls -l *.wasm
