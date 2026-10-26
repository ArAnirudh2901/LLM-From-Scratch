// Transformer forward pass for the browser (WebAssembly SIMD), exact fp32, same math as model_gpt.GPT.
// Rows (new tokens) attend to explicit lists of KV slots, so a beam search can share the story's keys/values
// and every chain token is computed once (see fast_qa.py, the Python reference).
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
typedef v128_t f4;
#define f4_load wasm_v128_load
#define f4_store wasm_v128_store
#define f4_splat wasm_f32x4_splat
#define f4_add wasm_f32x4_add
#define f4_mul wasm_f32x4_mul
static inline float f4_lane(f4 a, int i) {
  switch (i) { case 0: return wasm_f32x4_extract_lane(a, 0); case 1: return wasm_f32x4_extract_lane(a, 1);
               case 2: return wasm_f32x4_extract_lane(a, 2); default: return wasm_f32x4_extract_lane(a, 3); }
}
#else   // older CPUs / browsers without WebAssembly SIMD
typedef struct { float v[4]; } f4;
static inline f4 f4_load(const float *p) { f4 r = {{p[0], p[1], p[2], p[3]}}; return r; }
static inline void f4_store(float *p, f4 a) { p[0] = a.v[0]; p[1] = a.v[1]; p[2] = a.v[2]; p[3] = a.v[3]; }
static inline f4 f4_splat(float x) { f4 r = {{x, x, x, x}}; return r; }
static inline f4 f4_add(f4 a, f4 b) { f4 r = {{a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2], a.v[3] + b.v[3]}}; return r; }
static inline f4 f4_mul(f4 a, f4 b) { f4 r = {{a.v[0] * b.v[0], a.v[1] * b.v[1], a.v[2] * b.v[2], a.v[3] * b.v[3]}}; return r; }
static inline float f4_lane(f4 a, int i) { return a.v[i]; }
#endif

#define MAXL 16

typedef struct {
  int L, H, C, D, V, T, F, cap, rmax;
  const float *tok, *pos, *lnf_w, *lnf_b, *head;
  struct { const float *ln1_w, *ln1_b, *qkv, *proj, *ln2_w, *ln2_b, *fc1, *fc1_b, *fc2, *fc2_b; } l[MAXL];
  float *K, *Vv;                          // slot pool [L][cap][C]
  float *x, *xn, *qkv, *att, *h1, *sc;    // scratch
} Model;

#ifdef __wasm_simd128__
static inline f4 f4_min(f4 a, f4 b) { return wasm_f32x4_min(a, b); }
static inline f4 f4_max(f4 a, f4 b) { return wasm_f32x4_max(a, b); }
static inline f4 f4_div(f4 a, f4 b) { return wasm_f32x4_div(a, b); }
static inline f4 f4_round(f4 a) { return wasm_f32x4_nearest(a); }
static inline f4 f4_pow2i(f4 n) {   // 2^n for integral n in [-126, 127]
  return wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(n), wasm_i32x4_splat(127)), 23);
}
#else
static inline f4 f4_min(f4 a, f4 b) { f4 r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i]; return r; }
static inline f4 f4_max(f4 a, f4 b) { f4 r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i]; return r; }
static inline f4 f4_div(f4 a, f4 b) { f4 r; for (int i = 0; i < 4; i++) r.v[i] = a.v[i] / b.v[i]; return r; }
static inline f4 f4_round(f4 a) { f4 r; for (int i = 0; i < 4; i++) r.v[i] = nearbyintf(a.v[i]); return r; }
static inline f4 f4_pow2i(f4 n) {
  f4 r;
  for (int i = 0; i < 4; i++) { union { int i; float f; } u; u.i = ((int)n.v[i] + 127) << 23; r.v[i] = u.f; }
  return r;
}
#endif

// e^x, 4 lanes, ~1 ulp (Cephes expf): 2^n * p(r) with r = x - n ln2 split in two parts for accuracy.
static inline f4 f4_exp(f4 x) {
  x = f4_max(f4_min(x, f4_splat(88.3762626647949f)), f4_splat(-87.3365447505531f));
  f4 n = f4_round(f4_mul(x, f4_splat(1.44269504088896341f)));
  x = f4_add(x, f4_mul(n, f4_splat(-0.693359375f)));
  x = f4_add(x, f4_mul(n, f4_splat(2.12194440e-4f)));
  f4 p = f4_splat(1.9875691500E-4f);
  p = f4_add(f4_mul(p, x), f4_splat(1.3981999507E-3f));
  p = f4_add(f4_mul(p, x), f4_splat(8.3334519073E-3f));
  p = f4_add(f4_mul(p, x), f4_splat(4.1665795894E-2f));
  p = f4_add(f4_mul(p, x), f4_splat(1.6666665459E-1f));
  p = f4_add(f4_mul(p, x), f4_splat(5.0000001201E-1f));
  p = f4_add(f4_add(f4_mul(f4_mul(p, x), x), x), f4_splat(1.0f));
#ifdef __wasm_simd128__
  return wasm_f32x4_mul(p, f4_pow2i(n));
#else
  return f4_mul(p, f4_pow2i(n));
#endif
}

// erf(x), 4 lanes: rational approximation used by XLA/Eigen for GELU (no exp); |x| >= 4 gives +-1.
static inline f4 f4_erf(f4 x) {
  x = f4_max(f4_min(x, f4_splat(4.0f)), f4_splat(-4.0f));
  f4 x2 = f4_mul(x, x);
  f4 p = f4_splat(-2.72614225801306e-10f);
  p = f4_add(f4_mul(p, x2), f4_splat(2.77068142495902e-08f));
  p = f4_add(f4_mul(p, x2), f4_splat(-2.10102402082508e-06f));
  p = f4_add(f4_mul(p, x2), f4_splat(-5.69250639462346e-05f));
  p = f4_add(f4_mul(p, x2), f4_splat(-7.34990630326855e-04f));
  p = f4_add(f4_mul(p, x2), f4_splat(-2.95459980854025e-03f));
  p = f4_add(f4_mul(p, x2), f4_splat(-1.60960333262415e-02f));
  p = f4_mul(p, x);
  f4 q = f4_splat(-1.45660718464996e-05f);
  q = f4_add(f4_mul(q, x2), f4_splat(-2.13374055278905e-04f));
  q = f4_add(f4_mul(q, x2), f4_splat(-1.68282697438203e-03f));
  q = f4_add(f4_mul(q, x2), f4_splat(-7.37332916720468e-03f));
  q = f4_add(f4_mul(q, x2), f4_splat(-1.42647390514189e-02f));
  return f4_div(p, q);
}

#ifdef PROF   // build with -DPROF: per-section milliseconds, read with prof_get
#include <emscripten.h>
static double prof[8];
#define TIC double _t = emscripten_get_now();
#define TOC(i) { double _u = emscripten_get_now(); prof[i] += _u - _t; _t = _u; }
double *prof_get(void) { return prof; }
#else
#define TIC
#define TOC(i)
#endif

static inline float hsum(f4 a) { return f4_lane(a, 0) + f4_lane(a, 1) + f4_lane(a, 2) + f4_lane(a, 3); }

// a*b + c: one fused instruction in the relaxed-SIMD build, mul + add otherwise
#ifdef __wasm_relaxed_simd__
#define f4_madd(a, b, c) wasm_f32x4_relaxed_madd(a, b, c)
#else
#define f4_madd(a, b, c) f4_add(c, f4_mul(a, b))
#endif

// Y[R][N] = X[R][K] . W[N][K]^T (+ b); K % 4 == 0. Outputs outer, rows inner: each block of W is read from memory
// once and reused for every row while in L1. Tiles of RB rows x NB outputs; the best tile depends on the CPU's
// vector registers (16 on x86, 32 on ARM), so model_new times them and keeps the fastest.
#define TILE(RB, NB)                                                                                          \
  static void tile_##RB##x##NB(const float *X, const float *W, const float *b, float *Y, int r, int n, int N,  \
                               int K) {                                                                       \
    f4 acc[RB][NB];                                                                                           \
    for (int i = 0; i < RB; i++)                                                                              \
      for (int j = 0; j < NB; j++) acc[i][j] = f4_splat(0);                                                   \
    for (int k = 0; k < K; k += 4) {                                                                          \
      f4 w[NB];                                                                                               \
      for (int j = 0; j < NB; j++) w[j] = f4_load(W + (size_t)(n + j) * K + k);                              \
      for (int i = 0; i < RB; i++) {                                                                          \
        f4 x = f4_load(X + (size_t)(r + i) * K + k);                                                         \
        for (int j = 0; j < NB; j++) acc[i][j] = f4_madd(x, w[j], acc[i][j]);                                 \
      }                                                                                                       \
    }                                                                                                         \
    for (int i = 0; i < RB; i++)                                                                              \
      for (int j = 0; j < NB; j++) Y[(size_t)(r + i) * N + n + j] = hsum(acc[i][j]) + (b ? b[n + j] : 0);     \
  }
TILE(4, 4) TILE(2, 4) TILE(4, 2) TILE(6, 4) TILE(3, 4) TILE(1, 4) TILE(1, 1)

typedef void (*tile_fn)(const float *, const float *, const float *, float *, int, int, int, int);
static const struct { tile_fn f; int rb, nb; } TILES[] = {{tile_4x4, 4, 4}, {tile_2x4, 2, 4}, {tile_4x2, 4, 2},
                                                         {tile_6x4, 6, 4}, {tile_3x4, 3, 4}};
static int tile_pick = 0;

// One block column range [n0, n1) of the output, all rows, with the chosen tile.
static void gemm_cols(const float *X, const float *W, const float *b, float *Y, int R, int N, int K, int n0, int n1) {
  const tile_fn f = TILES[tile_pick].f;
  const int rb = TILES[tile_pick].rb, nb = TILES[tile_pick].nb, Rt = R - R % rb;
  int n = n0;
  for (; n + nb <= n1; n += nb) {
    for (int r = 0; r < Rt; r += rb) f(X, W, b, Y, r, n, N, K);
    for (int r = Rt; r < R; r++)
      for (int j = 0; j < nb; j += nb - j >= 4 ? 4 : 1) {
        if (nb - j >= 4) tile_1x4(X, W, b, Y, r, n + j, N, K); else tile_1x1(X, W, b, Y, r, n + j, N, K);
      }
  }
  for (; n < n1; n++)
    for (int r = 0; r < R; r++) tile_1x1(X, W, b, Y, r, n, N, K);
}

#ifdef __EMSCRIPTEN_PTHREADS__
// Thread pool: workers take chunks of output columns from a shared counter (balances fast and slow cores).
// Each output is computed by exactly one thread with the same arithmetic, so results do not depend on threads.
#include <pthread.h>
#include <stdatomic.h>
#include <emscripten/threading.h>
#define CHUNK 32
static int nthreads = 1, active = 1, keep = 16;   // workers started / used per GEMM / allowed to live
static struct { const float *X, *W, *b; float *Y; int R, N, K; } job;
static _Atomic int gen, busy, next_col;

static void run_chunks(void) {
  for (;;) {
    int n0 = atomic_fetch_add(&next_col, CHUNK);
    if (n0 >= job.N) return;
    int n1 = n0 + CHUNK < job.N ? n0 + CHUNK : job.N;
    gemm_cols(job.X, job.W, job.b, job.Y, job.R, job.N, job.K, n0, n1);
  }
}

static void *worker(void *arg) {
  const int id = (int)(intptr_t)arg;
  int seen = 0;
  for (;;) {
    for (int spins = 0; atomic_load(&gen) == seen; spins++)   // spin briefly between GEMMs, then sleep
      if (spins > 20000) emscripten_futex_wait((void *)&gen, seen, 2.0);
    seen = atomic_load(&gen);
    if (id >= keep) return 0;   // not chosen by tuning: exit instead of spinning on every step
    if (id < active) { run_chunks(); atomic_fetch_sub(&busy, 1); }
  }
  return 0;
}

static void gemm(const float *X, const float *W, const float *b, float *Y, int R, int N, int K);

// Start up to n - 1 workers (the caller is thread n), then keep the thread count that runs a typical step fastest
// (on CPUs with slow efficiency cores fewer threads can win). Returns the count used. scratch: >= 12 * 1536 floats.
int set_threads(int n, const float *W1, const float *W2, float *X, float *Y, int C, int F) {
  if (n > 16) n = 16;
  for (; nthreads < n; nthreads++) {
    pthread_t t;
    if (pthread_create(&t, 0, worker, (void *)(intptr_t)nthreads) != 0) break;
    pthread_detach(t);
  }
  for (int i = 0; i < 12 * F; i++) X[i] = (float)((i * 7919) % 101) / 101.0f - 0.5f;
  double best = 1e30;
  int pick = 1;
  active = nthreads;
  for (int rep = 0; rep < 20; rep++) gemm(X, W1, 0, Y, 12, F, C);   // warm up threads and CPU clocks
  for (int a = 1; a <= nthreads; a++) {
    active = a;
    double lo = 1e30;
    for (int rep = 0; rep < 13; rep++) {
      double t0 = emscripten_get_now();
      gemm(X, W1, 0, Y, 12, F, C);   // fc1-sized and fc2-sized products
      gemm(Y, W2, 0, X, 12, C, F);
      double dt = emscripten_get_now() - t0;
      if (rep >= 3 && dt < lo) lo = dt;
    }
    if (lo < best * 0.95) { best = lo; pick = a; }   // more threads only if clearly faster
  }
  active = pick;
  keep = pick;   // let the others exit: wake them with an empty job
  atomic_store(&next_col, 1 << 30);
  atomic_store(&busy, 0);
  atomic_fetch_add(&gen, 1);
  emscripten_futex_wake((void *)&gen, nthreads);
  nthreads = pick;
  return active;
}

static void gemm(const float *X, const float *W, const float *b, float *Y, int R, int N, int K) {
  if (active == 1 || (double)R * N * K < 400000.0) { gemm_cols(X, W, b, Y, R, N, K, 0, N); return; }
  job.X = X; job.W = W; job.b = b; job.Y = Y; job.R = R; job.N = N; job.K = K;
  atomic_store(&next_col, 0);
  atomic_store(&busy, active - 1);
  atomic_fetch_add(&gen, 1);
  emscripten_futex_wake((void *)&gen, active);
  run_chunks();
  while (atomic_load(&busy) > 0) {}   // workers finish their last chunk
}
#else
int set_threads(int n, const float *W1, const float *W2, float *X, float *Y, int C, int F) {
  (void)n; (void)W1; (void)W2; (void)X; (void)Y; (void)C; (void)F; return 1;
}
static void gemm(const float *X, const float *W, const float *b, float *Y, int R, int N, int K) {
  gemm_cols(X, W, b, Y, R, N, K, 0, N);
}
#endif

static void layernorm(const float *x, const float *w, const float *b, float *y, int R, int C) {
  for (int r = 0; r < R; r++) {
    const float *xr = x + r * C;
    float *yr = y + r * C;
    float mu = 0, var = 0;
    for (int i = 0; i < C; i++) mu += xr[i];
    mu /= C;
    for (int i = 0; i < C; i++) { float d = xr[i] - mu; var += d * d; }
    float inv = 1.0f / sqrtf(var / C + 1e-5f);
    for (int i = 0; i < C; i++) yr[i] = (xr[i] - mu) * inv * w[i] + b[i];
  }
}

#include <emscripten.h>
// Pick the fastest GEMM tile for this CPU: a few runs of a typical step (12 rows x first layer's qkv weights).
static int rmax_tune(Model *m) { return m->rmax < 12 ? m->rmax : 12; }
static void autotune(Model *m) {
  const int R = rmax_tune(m), N = 3 * m->C, K = m->C;
  for (int i = 0; i < R * K; i++) m->xn[i] = (float)((i * 7919) % 101) / 101.0f - 0.5f;
  double best = 1e30;
  int pick = 0;
  for (int t = 0; t < (int)(sizeof(TILES) / sizeof(TILES[0])); t++) {
    tile_pick = t;
    double lo = 1e30;
    for (int rep = 0; rep < 5; rep++) {
      double t0 = emscripten_get_now();
      gemm(m->xn, m->l[0].qkv, 0, m->qkv, R, N, K);
      double dt = emscripten_get_now() - t0;
      if (dt < lo) lo = dt;
    }
    if (lo < best) { best = lo; pick = t; }
  }
  tile_pick = pick;
}

int tile_choice(void) { return TILES[tile_pick].rb * 10 + TILES[tile_pick].nb; }

static float *take(const float **p, int n) { const float *q = *p; *p += n; return (float *)q; }

// weights: tok[V][C] pos[T][C], per layer ln1_w ln1_b qkv[3C][C] proj[C][C] ln2_w ln2_b fc1[4C][C] fc1_b fc2[C][4C] fc2_b,
// ln_f_w ln_f_b head[V][C]  (the order export_web.py writes)
Model *model_new(const float *w, int L, int H, int C, int V, int T, int cap, int rmax) {
  Model *m = calloc(1, sizeof(Model));
  m->L = L; m->H = H; m->C = C; m->D = C / H; m->V = V; m->T = T; m->F = 4 * C; m->cap = cap; m->rmax = rmax;
  m->tok = take(&w, V * C); m->pos = take(&w, T * C);
  for (int l = 0; l < L; l++) {
    m->l[l].ln1_w = take(&w, C); m->l[l].ln1_b = take(&w, C); m->l[l].qkv = take(&w, 3 * C * C);
    m->l[l].proj = take(&w, C * C); m->l[l].ln2_w = take(&w, C); m->l[l].ln2_b = take(&w, C);
    m->l[l].fc1 = take(&w, 4 * C * C); m->l[l].fc1_b = take(&w, 4 * C);
    m->l[l].fc2 = take(&w, 4 * C * C); m->l[l].fc2_b = take(&w, C);
  }
  m->lnf_w = take(&w, C); m->lnf_b = take(&w, C); m->head = take(&w, V * C);
  m->K = malloc(sizeof(float) * L * cap * C); m->Vv = malloc(sizeof(float) * L * cap * C);
  m->x = malloc(sizeof(float) * rmax * C); m->xn = malloc(sizeof(float) * rmax * C);
  m->qkv = malloc(sizeof(float) * rmax * 3 * C); m->att = malloc(sizeof(float) * rmax * C);
  m->h1 = malloc(sizeof(float) * rmax * 4 * C); m->sc = malloc(sizeof(float) * cap);
  if (!(m->K && m->Vv && m->x && m->h1)) return 0;
  autotune(m);
  return m;
}

// R new rows: token ids, positions, their slots, and the slots each row attends to (CSR: ctx[off[r]..off[r+1]),
// which must include the row's own slot). Writes ln_f(hidden) to out [R][C]. Returns 0, or -1 if R > rmax.
int forward(Model *m, int R, const int *tok, const int *pos, const int *slot, const int *off, const int *ctx, float *out) {
  if (R > m->rmax) return -1;
  const int C = m->C, D = m->D, H = m->H;
  const float scale = 1.0f / sqrtf((float)D);
  for (int r = 0; r < R; r++) {
    const float *a = m->tok + tok[r] * C, *b = m->pos + pos[r] * C;
    for (int i = 0; i < C; i++) m->x[r * C + i] = a[i] + b[i];
  }
  TIC
  for (int l = 0; l < m->L; l++) {
    float *Kl = m->K + (size_t)l * m->cap * C, *Vl = m->Vv + (size_t)l * m->cap * C;
    layernorm(m->x, m->l[l].ln1_w, m->l[l].ln1_b, m->xn, R, C);
    TOC(0)
    gemm(m->xn, m->l[l].qkv, 0, m->qkv, R, 3 * C, C);
    TOC(1)
    for (int r = 0; r < R; r++) {
      memcpy(Kl + (size_t)slot[r] * C, m->qkv + r * 3 * C + C, sizeof(float) * C);
      memcpy(Vl + (size_t)slot[r] * C, m->qkv + r * 3 * C + 2 * C, sizeof(float) * C);
    }
    for (int r = 0; r < R; r++) {
      const int *cs = ctx + off[r];
      const int n = off[r + 1] - off[r];
      for (int h = 0; h < H; h++) {
        const float *q = m->qkv + r * 3 * C + h * D;
        float mx = -INFINITY;
        for (int j = 0; j < n; j++) {
          const float *k = Kl + (size_t)cs[j] * C + h * D;
          f4 a = f4_splat(0);
          for (int d = 0; d < D; d += 4) a = f4_add(a, f4_mul(f4_load(q + d), f4_load(k + d)));
          float s = hsum(a) * scale;
          m->sc[j] = s;
          if (s > mx) mx = s;
        }
        float tot = 0;
        int j = 0;
        f4 mxv = f4_splat(mx), tv = f4_splat(0);
        for (; j + 4 <= n; j += 4) { f4 e = f4_exp(f4_add(f4_load(m->sc + j), f4_mul(mxv, f4_splat(-1.0f)))); f4_store(m->sc + j, e); tv = f4_add(tv, e); }
        tot = hsum(tv);
        for (; j < n; j++) {
          float tmp[4] = {m->sc[j] - mx, 0, 0, 0};
          m->sc[j] = f4_lane(f4_exp(f4_load(tmp)), 0); tot += m->sc[j];
        }
        float inv = 1.0f / tot;
        f4 acc[16];
        for (int d = 0; d < D / 4; d++) acc[d] = f4_splat(0);
        for (int j = 0; j < n; j++) {
          const float *v = Vl + (size_t)cs[j] * C + h * D;
          f4 p = f4_splat(m->sc[j] * inv);
          for (int d = 0; d < D / 4; d++) acc[d] = f4_add(acc[d], f4_mul(p, f4_load(v + 4 * d)));
        }
        for (int d = 0; d < D / 4; d++) f4_store(m->att + r * C + h * D + 4 * d, acc[d]);
      }
    }
    TOC(2)
    gemm(m->att, m->l[l].proj, 0, m->xn, R, C, C);
    for (int i = 0; i < R * C; i++) m->x[i] += m->xn[i];
    layernorm(m->x, m->l[l].ln2_w, m->l[l].ln2_b, m->xn, R, C);
    TOC(1)
    gemm(m->xn, m->l[l].fc1, m->l[l].fc1_b, m->h1, R, 4 * C, C);
    TOC(1)
    for (int i = 0; i < R * 4 * C; i += 4) {   // GELU, exact form 0.5 x (1 + erf(x / sqrt 2)); 4C % 4 == 0
      f4 v = f4_load(m->h1 + i);
      f4_store(m->h1 + i, f4_mul(f4_mul(f4_splat(0.5f), v), f4_add(f4_splat(1.0f), f4_erf(f4_mul(v, f4_splat(0.70710678118654752f))))));
    }
    TOC(3)
    gemm(m->h1, m->l[l].fc2, m->l[l].fc2_b, m->xn, R, C, 4 * C);
    TOC(1)
    for (int i = 0; i < R * C; i++) m->x[i] += m->xn[i];
    TOC(0)
  }
  layernorm(m->x, m->lnf_w, m->lnf_b, out, R, C);
  return 0;
}

// Full-vocabulary logits for n hidden rows: out[n][V]; lse[n] = log sum exp of each row (for log-softmax).
void logits(Model *m, const float *h, int n, float *out, float *lse) {
  TIC
  gemm(h, m->head, 0, out, n, m->V, m->C);
  TOC(4)
  if (!lse) return;
  for (int r = 0; r < n; r++) {
    const float *x = out + (size_t)r * m->V;
    float mx = -INFINITY;
    for (int i = 0; i < m->V; i++) if (x[i] > mx) mx = x[i];
    f4 mxv = f4_splat(-mx), tv = f4_splat(0);
    int i = 0;
    for (; i + 4 <= m->V; i += 4) tv = f4_add(tv, f4_exp(f4_add(f4_load(x + i), mxv)));
    float tot = hsum(tv);
    for (; i < m->V; i++) { float t[4] = {x[i] - mx, 0, 0, 0}; tot += f4_lane(f4_exp(f4_load(t)), 0); }
    lse[r] = mx + logf(tot);
  }
  TOC(5)
}

// Logits for k chosen vocabulary ids only: out[n][k].
void logits_some(Model *m, const float *h, int n, const int *ids, int k, float *out) {
  for (int r = 0; r < n; r++)
    for (int j = 0; j < k; j++) {
      const float *w = m->head + (size_t)ids[j] * m->C, *x = h + r * m->C;
      f4 a = f4_splat(0);
      for (int i = 0; i < m->C; i += 4) a = f4_add(a, f4_mul(f4_load(x + i), f4_load(w + i)));
      out[r * k + j] = hsum(a);
    }
}

int model_cap(Model *m) { return m->cap; }

// CRC-32 (IEEE, same as zlib.crc32), slicing-by-8: checks the weights in memory against the value export_web.py
// recorded, so a corrupted, truncated or mismatched weights file is refused instead of giving wrong answers.
static unsigned crc_t[8][256];
unsigned crc32(const unsigned char *p, unsigned n, unsigned crc) {
  if (!crc_t[0][1]) {
    for (unsigned i = 0; i < 256; i++) {
      unsigned c = i;
      for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      crc_t[0][i] = c;
    }
    for (unsigned i = 0; i < 256; i++)
      for (int t = 1; t < 8; t++) crc_t[t][i] = (crc_t[t - 1][i] >> 8) ^ crc_t[0][crc_t[t - 1][i] & 0xFF];
  }
  crc = ~crc;
  for (; n >= 8; p += 8, n -= 8) {
    unsigned a = crc ^ (p[0] | p[1] << 8 | p[2] << 16 | (unsigned)p[3] << 24), b = p[4] | p[5] << 8 | p[6] << 16 | (unsigned)p[7] << 24;
    crc = crc_t[7][a & 0xFF] ^ crc_t[6][(a >> 8) & 0xFF] ^ crc_t[5][(a >> 16) & 0xFF] ^ crc_t[4][a >> 24] ^
          crc_t[3][b & 0xFF] ^ crc_t[2][(b >> 8) & 0xFF] ^ crc_t[1][(b >> 16) & 0xFF] ^ crc_t[0][b >> 24];
  }
  while (n--) crc = crc_t[0][(crc ^ *p++) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

int model_set_threads(Model *m, int n) {   // tune threads on this model's own first-layer FFN weights
  return set_threads(n, m->l[0].fc1, m->l[0].fc2, m->h1, m->xn, m->C, 4 * m->C);
}

// Grow the KV slot pool to newcap slots, keeping the first `used` slots. Returns 0, or -1 when out of memory.
int model_grow(Model *m, int newcap, int used) {
  if (newcap <= m->cap) return 0;
  float *K = malloc(sizeof(float) * m->L * newcap * m->C), *V = malloc(sizeof(float) * m->L * newcap * m->C),
        *sc = malloc(sizeof(float) * newcap);
  if (!K || !V || !sc) { free(K); free(V); free(sc); return -1; }
  for (int l = 0; l < m->L; l++) {
    memcpy(K + (size_t)l * newcap * m->C, m->K + (size_t)l * m->cap * m->C, sizeof(float) * used * m->C);
    memcpy(V + (size_t)l * newcap * m->C, m->Vv + (size_t)l * m->cap * m->C, sizeof(float) * used * m->C);
  }
  free(m->K); free(m->Vv); free(m->sc);
  m->K = K; m->Vv = V; m->sc = sc; m->cap = newcap;
  return 0;
}
