// The v10 reader forward pass reuses the fp32 SIMD GEMM kernels from engine.c. The existing v8 ABI stays intact.
#include "engine.c"

typedef struct {
  const float *ln1, *ln2, *qkv, *proj, *qn, *kn, *w1, *w3, *w2;
} ReaderLayer;

typedef struct {
  int L, H, C, D, V, T, F, DC, cap, rmax;
  float eps, rope_base;
  const float *tok, *lnf, *copy_q, *copy_k, *gate_w, *gate_b;
  ReaderLayer l[MAXL];
  float *K, *Vv, *x, *xn, *qkv, *att, *h1, *h2, *sc;
} Reader;

static void *take_reader(const float **p, size_t n) {
  const float *q = *p;
  *p += n;
  return (void *)q;
}

static void rmsnorm_rows(const float *x, const float *w, float *y, int R, int C, float eps) {
  for (int r = 0; r < R; r++) {
    const float *xr = x + (size_t)r * C;
    float *yr = y + (size_t)r * C;
    float ss = 0.0f;
    for (int i = 0; i < C; i++) ss += xr[i] * xr[i];
    float inv = 1.0f / sqrtf(ss / C + eps);
    for (int i = 0; i < C; i++) yr[i] = xr[i] * inv * w[i];
  }
}

static void rmsnorm_heads(float *x, int R, int H, int D, const float *w, float eps) {
  for (int r = 0; r < R; r++) for (int h = 0; h < H; h++) {
    float *v = x + (size_t)r * H * D + h * D;
    float ss = 0.0f;
    for (int i = 0; i < D; i++) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss / D + eps);
    for (int i = 0; i < D; i++) v[i] = v[i] * inv * w[i];
  }
}

static void rope_rows(float *x, int R, int H, int D, const int *pos, double base) {
  for (int r = 0; r < R; r++) for (int h = 0; h < H; h++) {
    float *v = x + (size_t)r * H * D + h * D;
    for (int i = 0; i < D; i += 2) {
      double inv = 1.0 / pow(base, (double)i / D);
      double a = (double)pos[r] * inv;
      float c = (float)cos(a), s = (float)sin(a), x0 = v[i], x1 = v[i + 1];
      v[i] = x0 * c - x1 * s;
      v[i + 1] = x0 * s + x1 * c;
    }
  }
}

Reader *reader_new(const float *weights, int L, int H, int C, int V, int T, int F, int DC, int cap, int rmax,
                   float eps, double rope_base) {
  if (L < 1 || L > MAXL || H < 1 || C % H || C / H % 4 || F % 4 || DC < 1 || cap < 1 || rmax < 1) return 0;
  Reader *r = calloc(1, sizeof(Reader));
  if (!r) return 0;
  r->L = L; r->H = H; r->C = C; r->D = C / H; r->V = V; r->T = T; r->F = F; r->DC = DC;
  r->cap = cap; r->rmax = rmax; r->eps = eps; r->rope_base = rope_base;
  r->tok = take_reader(&weights, (size_t)V * C);
  for (int i = 0; i < L; i++) {
    ReaderLayer *b = &r->l[i];
    b->ln1 = take_reader(&weights, C); b->ln2 = take_reader(&weights, C);
    b->qkv = take_reader(&weights, (size_t)3 * C * C); b->proj = take_reader(&weights, (size_t)C * C);
    b->qn = take_reader(&weights, r->D); b->kn = take_reader(&weights, r->D);
    b->w1 = take_reader(&weights, (size_t)F * C); b->w3 = take_reader(&weights, (size_t)F * C);
    b->w2 = take_reader(&weights, (size_t)C * F);
  }
  r->lnf = take_reader(&weights, C);
  r->copy_q = take_reader(&weights, (size_t)DC * C); r->copy_k = take_reader(&weights, (size_t)DC * C);
  r->gate_w = take_reader(&weights, C); r->gate_b = take_reader(&weights, 1);
  r->K = malloc(sizeof(float) * (size_t)L * cap * C); r->Vv = malloc(sizeof(float) * (size_t)L * cap * C);
  r->x = malloc(sizeof(float) * (size_t)rmax * C); r->xn = malloc(sizeof(float) * (size_t)rmax * C);
  r->qkv = malloc(sizeof(float) * (size_t)rmax * 3 * C); r->att = malloc(sizeof(float) * (size_t)rmax * C);
  r->h1 = malloc(sizeof(float) * (size_t)rmax * F); r->h2 = malloc(sizeof(float) * (size_t)rmax * F);
  r->sc = malloc(sizeof(float) * (size_t)cap);
  if (!(r->K && r->Vv && r->x && r->xn && r->qkv && r->att && r->h1 && r->h2 && r->sc)) return 0;
  Model tune = {0};
  tune.C = C; tune.rmax = 12; tune.xn = r->xn; tune.qkv = r->qkv; tune.l[0].qkv = r->l[0].qkv;
  autotune(&tune);
  return r;
}

int reader_grow(Reader *r, int newcap, int used) {
  if (newcap <= r->cap) return 0;
  float *K = malloc(sizeof(float) * (size_t)r->L * newcap * r->C);
  float *Vv = malloc(sizeof(float) * (size_t)r->L * newcap * r->C);
  float *sc = malloc(sizeof(float) * (size_t)newcap);
  if (!K || !Vv || !sc) { free(K); free(Vv); free(sc); return -1; }
  for (int l = 0; l < r->L; l++) {
    memcpy(K + (size_t)l * newcap * r->C, r->K + (size_t)l * r->cap * r->C, sizeof(float) * (size_t)used * r->C);
    memcpy(Vv + (size_t)l * newcap * r->C, r->Vv + (size_t)l * r->cap * r->C, sizeof(float) * (size_t)used * r->C);
  }
  free(r->K); free(r->Vv); free(r->sc);
  r->K = K; r->Vv = Vv; r->sc = sc; r->cap = newcap;
  return 0;
}

// CSR context lists let the caller share prefix and beam-parent KV slots. Every row writes its KV at slot[row].
int reader_forward(Reader *m, int R, const int *tok, const int *pos, const int *slot,
                   const int *off, const int *ctx, float *out) {
  if (R > m->rmax) return -1;
  const int C = m->C, D = m->D, H = m->H, F = m->F;
  const float scale = 1.0f / sqrtf((float)D);
  for (int r = 0; r < R; r++) memcpy(m->x + (size_t)r * C, m->tok + (size_t)tok[r] * C, sizeof(float) * C);
  for (int layer = 0; layer < m->L; layer++) {
    ReaderLayer *b = &m->l[layer];
    float *Kl = m->K + (size_t)layer * m->cap * C, *Vl = m->Vv + (size_t)layer * m->cap * C;
    rmsnorm_rows(m->x, b->ln1, m->xn, R, C, m->eps);
    gemm(m->xn, b->qkv, 0, m->qkv, R, 3 * C, C);
    for (int row = 0; row < R; row++) {
      float *q = m->qkv + (size_t)row * 3 * C;
      rmsnorm_heads(q, 1, H, D, b->qn, m->eps);
      rmsnorm_heads(q + C, 1, H, D, b->kn, m->eps);
      rope_rows(q, 1, H, D, pos + row, m->rope_base);
      rope_rows(q + C, 1, H, D, pos + row, m->rope_base);
      memcpy(Kl + (size_t)slot[row] * C, q + C, sizeof(float) * C);
      memcpy(Vl + (size_t)slot[row] * C, q + 2 * C, sizeof(float) * C);
    }
    for (int row = 0; row < R; row++) {
      const int *cs = ctx + off[row];
      const int n = off[row + 1] - off[row];
      for (int h = 0; h < H; h++) {
        const float *q = m->qkv + (size_t)row * 3 * C + h * D;
        float mx = -INFINITY;
        for (int j = 0; j < n; j++) {
          const float *k = Kl + (size_t)cs[j] * C + h * D;
          f4 a = f4_splat(0);
          for (int d = 0; d < D; d += 4) a = f4_add(a, f4_mul(f4_load(q + d), f4_load(k + d)));
          float s = hsum(a) * scale; m->sc[j] = s; if (s > mx) mx = s;
        }
        f4 mxv = f4_splat(-mx), total = f4_splat(0);
        int j = 0;
        for (; j + 4 <= n; j += 4) {
          f4 e = f4_exp(f4_add(f4_load(m->sc + j), mxv)); f4_store(m->sc + j, e); total = f4_add(total, e);
        }
        float sum = hsum(total);
        for (; j < n; j++) { float z[4] = {m->sc[j] - mx, 0, 0, 0}; m->sc[j] = f4_lane(f4_exp(f4_load(z)), 0); sum += m->sc[j]; }
        float inv = 1.0f / sum;
        f4 acc[16];
        for (int d = 0; d < D / 4; d++) acc[d] = f4_splat(0);
        for (int j2 = 0; j2 < n; j2++) {
          const float *v = Vl + (size_t)cs[j2] * C + h * D; f4 p = f4_splat(m->sc[j2] * inv);
          for (int d = 0; d < D / 4; d++) acc[d] = f4_add(acc[d], f4_mul(p, f4_load(v + 4 * d)));
        }
        for (int d = 0; d < D / 4; d++) f4_store(m->att + (size_t)row * C + h * D + 4 * d, acc[d]);
      }
    }
    gemm(m->att, b->proj, 0, m->xn, R, C, C);
    for (int i = 0; i < R * C; i++) m->x[i] += m->xn[i];
    rmsnorm_rows(m->x, b->ln2, m->xn, R, C, m->eps);
    gemm(m->xn, b->w1, 0, m->h1, R, F, C);
    gemm(m->xn, b->w3, 0, m->h2, R, F, C);
    for (int i = 0; i < R * F; i++) { float a = m->h1[i]; m->h1[i] = (a / (1.0f + expf(-a))) * m->h2[i]; }
    gemm(m->h1, b->w2, 0, m->xn, R, C, F);
    for (int i = 0; i < R * C; i++) m->x[i] += m->xn[i];
  }
  rmsnorm_rows(m->x, m->lnf, out, R, C, m->eps);
  return 0;
}

void reader_logits(Reader *m, const float *h, int n, float *out) {
  gemm(h, m->tok, 0, out, n, m->V, m->C);
}

void reader_copy_keys(Reader *m, const float *h, int n, float *out) {
  gemm(h, m->copy_k, 0, out, n, m->DC, m->C);
}

// Return vocab-plus-pointer log probabilities, matching Reader._mix (including its 1e-9 floor).
void reader_mix(Reader *m, const float *h, const float *logits, const float *keys,
                const int *ids, int nids, float *out) {
  if (nids <= 0) {
    float mx = -INFINITY;
    for (int i = 0; i < m->V; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0;
    for (int i = 0; i < m->V; i++) sum += exp((double)logits[i] - mx);
    double lse = mx + log(sum);
    for (int i = 0; i < m->V; i++) out[i] = logf((float)exp((double)logits[i] - lse) + 1e-9f);
    return;
  }
  float *q = malloc(sizeof(float) * m->DC), *scores = malloc(sizeof(float) * (size_t)nids);
  if (!q || !scores) { free(q); free(scores); return; }
  gemm(h, m->copy_q, 0, q, 1, m->DC, m->C);
  float mx = -INFINITY;
  for (int i = 0; i < m->V; i++) if (logits[i] > mx) mx = logits[i];
  double sum = 0;
  for (int i = 0; i < m->V; i++) sum += exp((double)logits[i] - mx);
  const double lse = mx + log(sum);
  double copy_max = -INFINITY;
  for (int j = 0; j < nids; j++) {
    double acc = 0;
    for (int d = 0; d < m->DC; d++) acc += (double)q[d] * keys[(size_t)j * m->DC + d];
    scores[j] = (float)(acc / sqrt((double)m->DC));
    if (scores[j] > copy_max) copy_max = scores[j];
  }
  double copy_sum = 0;
  for (int j = 0; j < nids; j++) copy_sum += exp((double)scores[j] - copy_max);
  double dot = 0;
  for (int i = 0; i < m->C; i++) dot += (double)h[i] * m->gate_w[i];
  float gate = 1.0f / (1.0f + expf(-(float)dot - m->gate_b[0]));
  for (int i = 0; i < m->V; i++) out[i] = gate * (float)exp((double)logits[i] - lse);
  for (int j = 0; j < nids; j++) out[ids[j]] += (1.0f - gate) * (float)(exp((double)scores[j] - copy_max) / copy_sum);
  for (int i = 0; i < m->V; i++) out[i] = logf(out[i] + 1e-9f);
  free(q); free(scores);
}
