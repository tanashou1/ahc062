use std::io::{self, Read};
use std::time::Instant;

const TIME_LIMIT_MS: u64 = 2950;

type Pos = (usize, usize);

struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed | 1 } }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_usize(&mut self, n: usize) -> usize { (self.next_u64() % n as u64) as usize }
    fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) }
}

fn main() {
    let timer = Instant::now();
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut iter = input.split_ascii_whitespace();

    let n: usize = iter.next().unwrap().parse().unwrap();
    let mut a = vec![vec![0i64; n]; n];
    for row in &mut a {
        for x in row { *x = iter.next().unwrap().parse().unwrap(); }
    }

    let mut path = snake_path(n);
    let raw = raw_score(&a, &path);
    eprintln!("init_score={}", display_score(raw, (n * n) as i64));

    let phase1_end = 150u64;
    sa_block_swap(&a, &mut path, n, &mut Rng::new(42), timer, phase1_end);
    sa_twoopt(&a, &mut path, n, &mut Rng::new(43), timer);

    validate_path(&path, n);
    let raw = raw_score(&a, &path);
    eprintln!("score={}", display_score(raw, (n * n) as i64));
    for (r, c) in path { println!("{r} {c}"); }
}

fn snake_path(n: usize) -> Vec<Pos> {
    let mut path = Vec::with_capacity(n * n);
    for i in 0..n {
        if i % 2 == 0 {
            for j in 0..n { path.push((i, j)); }
        } else {
            for j in (0..n).rev() { path.push((i, j)); }
        }
    }
    path
}

fn sa_block_swap(a: &[Vec<i64>], path: &mut Vec<Pos>, n: usize, rng: &mut Rng, timer: Instant, end_ms: u64) {
    let n2   = path.len();
    let rows = a.len();
    let cols = a[0].len();
    let max_k = n;

    let mut pos_in_path = vec![vec![0usize; cols]; rows];
    for (t, &(r, c)) in path.iter().enumerate() { pos_in_path[r][c] = t; }
    let mut a_val: Vec<i64> = path.iter().map(|&(r, c)| a[r][c]).collect();

    let sa_start_ms = timer.elapsed().as_millis() as f64;
    let sa_end_ms   = end_ms as f64;
    let sa_duration = (sa_end_ms - sa_start_ms).max(1.0);
    let t_start = 5e7f64;
    let t_end   = 1e3f64;
    let mut bs_iters    = 0u64;
    let mut bs_accepted = 0u64;
    let mut fo_iters    = 0u64;
    let mut fo_accepted = 0u64;
    let mut buf = vec![(0usize, 0usize); max_k];

    loop {
        let elapsed = timer.elapsed().as_millis() as f64;
        if elapsed >= sa_end_ms { break; }
        let progress = (elapsed - sa_start_ms) / sa_duration;
        let temp = t_start * (t_end / t_start).powf(progress);

        if rng.next_u64() % 2 == 0 {
            // ── 隣接グリッド行 4-opt (50%) ──────────────────────────────────
            let r1 = rng.next_usize(n - 1);
            let r2 = r1 + 1;
            let a_col = rng.next_usize(n);
            let k = 1 + rng.next_usize((n - a_col).min(max_k));

            let l1 = pos_in_path[r1][a_col];
            let dir1: i64 = if k > 1 {
                let nxt = pos_in_path[r1][a_col + 1];
                if nxt == l1 + 1 { 1 } else if nxt + 1 == l1 { -1 } else { continue }
            } else { 1 };
            let mut ok = true;
            for i in 2..k {
                let exp = l1 as i64 + dir1 * i as i64;
                if exp < 0 || pos_in_path[r1][a_col + i] as i64 != exp { ok = false; break; }
            }
            if !ok { continue; }
            let (la, ma) = if dir1 > 0 { (l1, l1 + k - 1) } else { (l1 + 1 - k, l1) };

            let l2 = pos_in_path[r2][a_col];
            let dir2: i64 = if k > 1 {
                let nxt = pos_in_path[r2][a_col + 1];
                if nxt == l2 + 1 { 1 } else if nxt + 1 == l2 { -1 } else { continue }
            } else { 1 };
            let mut ok = true;
            for i in 2..k {
                let exp = l2 as i64 + dir2 * i as i64;
                if exp < 0 || pos_in_path[r2][a_col + i] as i64 != exp { ok = false; break; }
            }
            if !ok { continue; }
            let (lb, qb) = if dir2 > 0 { (l2, l2 + k - 1) } else { (l2 + 1 - k, l2) };

            let (l, m, p, q) = if la < lb { (la, ma, lb, qb) } else { (lb, qb, la, ma) };
            if p <= m || l == 0 || q + 1 >= n2 { continue; }
            if !king_adj(path[l - 1], path[q]) { continue; }
            if !king_adj(path[l],     path[q + 1]) { continue; }
            if !king_adj(path[p],     path[m + 1]) { continue; }
            if !king_adj(path[p - 1], path[m]) { continue; }

            fo_iters += 1;
            let mut delta: i64 = 0;
            for i in 0..k {
                delta += (l + i) as i64 * (a_val[p + k - 1 - i] - a_val[l + i]);
                delta += (p + i) as i64 * (a_val[l + k - 1 - i] - a_val[p + i]);
            }
            if delta >= 0 || rng.next_f64() < (delta as f64 / temp).exp() {
                buf[..k].copy_from_slice(&path[l..=m]);
                for i in 0..k {
                    path[l + i] = path[p + k - 1 - i];
                    a_val[l + i] = a_val[p + k - 1 - i];
                }
                for i in 0..k {
                    path[p + i] = buf[k - 1 - i];
                    a_val[p + i] = a[buf[k - 1 - i].0][buf[k - 1 - i].1];
                }
                for i in l..=q { pos_in_path[path[i].0][path[i].1] = i; }
                fo_accepted += 1;
            }
            continue;
        }

        // ── 通常 block_swap ──────────────────────────────────────────────────
        let l = 1 + rng.next_usize(n2 - 2);
        let (pr, pc) = path[l - 1];
        for dr in -1i64..=1 {
            for dc in -1i64..=1 {
                if dr == 0 && dc == 0 { continue; }
                let nr = pr as i64 + dr;
                let nc = pc as i64 + dc;
                if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 { continue; }
                let q = pos_in_path[nr as usize][nc as usize];
                if q + 1 >= n2 || q <= l { continue; }
                if !king_adj(path[l], path[q + 1]) { continue; }

                let k_max = q.min(n2 - 1 - l).min((q - l) / 2);
                if k_max == 0 { continue; }
                let k = 1 + rng.next_usize(k_max.min(max_k));
                let m = l + k - 1;
                let p = q - k + 1;

                if !king_adj(path[p], path[m + 1]) { continue; }
                if !king_adj(path[p - 1], path[m]) { continue; }

                let mut delta: i64 = 0;
                for i in 0..k {
                    delta += (l + i) as i64 * (a_val[p + k - 1 - i] - a_val[l + i]);
                    delta += (p + i) as i64 * (a_val[l + k - 1 - i] - a_val[p + i]);
                }

                if delta >= 0 || rng.next_f64() < (delta as f64 / temp).exp() {
                    buf[..k].copy_from_slice(&path[l..=m]);
                    for i in 0..k {
                        path[l + i] = path[p + k - 1 - i];
                        a_val[l + i] = a_val[p + k - 1 - i];
                    }
                    for i in 0..k {
                        path[p + i] = buf[k - 1 - i];
                        a_val[p + i] = a[buf[k-1-i].0][buf[k-1-i].1];
                    }
                    for i in l..=q { pos_in_path[path[i].0][path[i].1] = i; }
                    bs_accepted += 1;
                }
                bs_iters += 1;
            }
        }
    }
    eprintln!("block_swap: iters={bs_iters} acc={bs_accepted}");
    eprintln!("fouropt:    iters={fo_iters} acc={fo_accepted}");
}

// ─────────────────────────────────────────────────────────────────────────────
// SA: 2-opt + or-opt 統合
//
// [2-opt] path[l..=r] を逆転 (50%の確率)
//   条件1: king_adj(path[l-1], path[r]) ← king隣接探索で保証
//   条件2: king_adj(path[l], path[r+1])  (r+1 < n2)
//   delta = Σ_{j=l}^{r} (l+r-2j) * a_val[j]
//
// [or-opt 右] [l..=m] を q > m の後ろへ移動
//   条件1: king_adj(path[l-1], path[m+1]) ← outer loop 保証
//   条件2: king_adj(path[q], path[l])     ← inner loop 保証
//   条件3: king_adj(path[m], path[q+1])  (q+1 < n2)
//   中間 [m+1..q] が左にずれる (shift -k)
//   delta = (q-m)*sum_seg1 - k*sum_inter_right
//
// [or-opt 左] [l..=m] を q < l-1 の後ろへ移動 (q+1 の前へ)
//   条件1: king_adj(path[l-1], path[m+1]) ← outer loop 保証
//   条件2: king_adj(path[q], path[l])     ← inner loop 保証
//   条件3: king_adj(path[m], path[q+1])
//   中間 [q+1..l-1] が右にずれる (shift +k)
//   delta = k*sum_inter_left - d*sum_seg1   (d = l-1-q)
// ─────────────────────────────────────────────────────────────────────────────
fn sa_twoopt(a: &[Vec<i64>], path: &mut Vec<Pos>, n: usize, rng: &mut Rng, timer: Instant) {
    let n2   = path.len();
    let rows = a.len();
    let cols = a[0].len();
    let oropt_max_dist = n2; // or-opt で q-m の最大距離 (unlimited)
    let max_k = n;              // or-opt で移動するセグメントの最大長

    let mut pos_in_path = vec![vec![0usize; cols]; rows];
    for (t, &(r, c)) in path.iter().enumerate() { pos_in_path[r][c] = t; }
    let mut a_val: Vec<i64> = path.iter().map(|&(r, c)| a[r][c]).collect();

    // Prefix sum: psum[i] = Σa_val[0..i]
    let mut psum = vec![0i64; n2 + 1];
    for i in 0..n2 { psum[i + 1] = psum[i] + a_val[i]; }

    // Weighted prefix sum: wsum[i] = Σ_{j<i} j * a_val[j]  (for O(1) 2-opt delta)
    let mut wsum = vec![0i64; n2 + 1];
    for i in 0..n2 { wsum[i + 1] = wsum[i] + i as i64 * a_val[i]; }

    let mut buf = vec![(0usize, 0usize); max_k];

    let sa_start_ms = timer.elapsed().as_millis() as f64;
    let sa_end_ms   = TIME_LIMIT_MS as f64;
    let sa_duration = (sa_end_ms - sa_start_ms).max(1.0);
    let t_start = 3e7f64;
    let t_end   = 3e4f64;
    let log_ratio = (t_end / t_start).ln();
    // Iterated SA: restart from best at midpoint with moderate temperature
    let t_start2   = 3e5f64;
    let log_ratio2 = (t_end / t_start2).ln();
    let midpoint_ms = sa_start_ms + sa_duration * 0.4;
    let mut best_raw: i64 = wsum[n2];
    let mut best_path_saved: Vec<Pos> = path.clone();
    let mut best_a_val_saved: Vec<i64> = a_val.clone();
    let mut restarted = false;
    let mut twoopt_iters    = 0u64;
    let mut twoopt_accepted = 0u64;
    let mut oropt_iters    = 0u64;
    let mut oropt_accepted = 0u64;

    let mut temp = t_start;
    let mut iter_count = 0u32;
    loop {
        // Update temperature every 256 iters to avoid per-iter powf/elapsed overhead
        if iter_count & 255 == 0 {
            let elapsed = timer.elapsed().as_millis() as f64;
            if elapsed >= sa_end_ms { break; }

            // Track best solution
            let cur_raw = wsum[n2];
            if cur_raw > best_raw {
                best_raw = cur_raw;
                best_path_saved.copy_from_slice(path);
                best_a_val_saved.copy_from_slice(&a_val);
            }

            // Restart at midpoint: restore best and reheat
            if !restarted && elapsed >= midpoint_ms {
                restarted = true;
                if best_raw > wsum[n2] {
                    path.copy_from_slice(&best_path_saved);
                    a_val.copy_from_slice(&best_a_val_saved);
                    for (t, &(r, c)) in path.iter().enumerate() { pos_in_path[r][c] = t; }
                    psum[0] = 0; wsum[0] = 0;
                    for i in 0..n2 {
                        psum[i + 1] = psum[i] + a_val[i];
                        wsum[i + 1] = wsum[i] + i as i64 * a_val[i];
                    }
                }
            }

            temp = if !restarted {
                let progress = (elapsed - sa_start_ms) / (midpoint_ms - sa_start_ms);
                t_start * (log_ratio * progress.min(1.0)).exp()
            } else {
                let progress = (elapsed - midpoint_ms) / (sa_end_ms - midpoint_ms);
                t_start2 * (log_ratio2 * progress.min(1.0)).exp()
            };
        }
        iter_count = iter_count.wrapping_add(1);

        let l = 1 + rng.next_usize(n2 - 2);

        let roll = rng.next_u64() % 16;
        if roll == 0 {  // 6.25% 2-opt
            // ── 2-opt ────────────────────────────────────────────────────────
            let (pr, pc) = path[l - 1];
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if dr == 0 && dc == 0 { continue; }
                    let nr = pr as i64 + dr;
                    let nc = pc as i64 + dc;
                    if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 { continue; }
                    let r = pos_in_path[nr as usize][nc as usize];
                    if r <= l { continue; }
                    if r + 1 < n2 && !king_adj(path[l], path[r + 1]) { continue; }

                    let lr_sum = (l + r) as i64;
                    let sum_seg = psum[r + 1] - psum[l];
                    let wseg = wsum[r + 1] - wsum[l];
                    let delta = lr_sum * sum_seg - 2 * wseg;

                    let accept = delta >= 0 || { let r = delta as f64 / temp; r > -30.0 && rng.next_f64() < r.exp() };
                    if accept {
                        path[l..=r].reverse();
                        a_val[l..=r].reverse();
                        for i in l..=r {
                            pos_in_path[path[i].0][path[i].1] = i;
                            psum[i + 1] = psum[i] + a_val[i];
                            wsum[i + 1] = wsum[i] + i as i64 * a_val[i];
                        }
                        twoopt_accepted += 1;
                    }
                    twoopt_iters += 1;
                }
            }
        } else if roll <= 5 {
            // ── block_swap: [l..m] ↔ [p..q] (equal size, both reversed) ─────
            let (pr, pc) = path[l - 1];
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if dr == 0 && dc == 0 { continue; }
                    let nr = pr as i64 + dr;
                    let nc = pc as i64 + dc;
                    if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 { continue; }
                    let q = pos_in_path[nr as usize][nc as usize];
                    if q + 1 >= n2 || q <= l { continue; }
                    if !king_adj(path[l], path[q + 1]) { continue; }

                    let k_max = q.min(n2 - 1 - l).min((q - l) / 2).min(3);
                    if k_max == 0 { continue; }
                    // Try all k in 1..=k_max, pick best delta
                    let mut best_k = 0usize;
                    let mut best_delta = i64::MIN;
                    for k in 1..=k_max {
                        let m = l + k - 1;
                        let p = q - k + 1;
                        if !king_adj(path[p], path[m + 1]) { continue; }
                        if !king_adj(path[p - 1], path[m]) { continue; }
                        let mut delta: i64 = 0;
                        for i in 0..k {
                            delta += (l + i) as i64 * (a_val[p + k - 1 - i] - a_val[l + i]);
                            delta += (p + i) as i64 * (a_val[l + k - 1 - i] - a_val[p + i]);
                        }
                        if delta > best_delta { best_delta = delta; best_k = k; }
                    }
                    if best_k == 0 { break; }
                    let k = best_k;
                    let delta = best_delta;
                    let m = l + k - 1;
                    let p = q - k + 1;

                    let accept = delta >= 0 || { let r = delta as f64 / temp; r > -30.0 && rng.next_f64() < r.exp() };
                    if accept {
                        buf[..k].copy_from_slice(&path[l..=m]);
                        for i in 0..k {
                            path[l + i] = path[p + k - 1 - i];
                            a_val[l + i] = a_val[p + k - 1 - i];
                        }
                        for i in 0..k {
                            path[p + i] = buf[k - 1 - i];
                            a_val[p + i] = a[buf[k - 1 - i].0][buf[k - 1 - i].1];
                        }
                        for i in l..=q {
                            pos_in_path[path[i].0][path[i].1] = i;
                            psum[i + 1] = psum[i] + a_val[i];
                            wsum[i + 1] = wsum[i] + i as i64 * a_val[i];
                        }
                    }
                    break;
                }
            }
        } else if roll <= 10 {
            // ── or-opt-rev: [l..=m] を reversed で p の後ろへ移動 (左右) ──
            // 条件1: king_adj(path[l-1], path[m+1]) ← outer loop 保証
            // 条件2: king_adj(path[p], path[m])    ← inner loop (path[m] の隣接)
            // 条件3: king_adj(path[l], path[p+1])  ← explicit check
            // 右: delta = -k*sum_C + (d-k+1+2m)*sum_B - 2*wseg_B  (d=p-m)
            // 左: delta = k*sum_F + (-d_l-k+1+2m)*sum_B - 2*wseg_B (d_l=l-1-p)
            let (pr, pc) = path[l - 1];
            'orrev: for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if dr == 0 && dc == 0 { continue; }
                    let nr = pr as i64 + dr;
                    let nc = pc as i64 + dc;
                    if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 { continue; }
                    let mp1 = pos_in_path[nr as usize][nc as usize];
                    if mp1 <= l || mp1 >= n2 { continue; }
                    let k = mp1 - l;
                    if k > max_k { continue; }
                    let m = mp1 - 1;

                    let (mr, mc) = path[m];
                    for dr2 in -1i64..=1 {
                        for dc2 in -1i64..=1 {
                            if dr2 == 0 && dc2 == 0 { continue; }
                            let nr2 = mr as i64 + dr2;
                            let nc2 = mc as i64 + dc2;
                            if nr2 < 0 || nr2 >= rows as i64 || nc2 < 0 || nc2 >= cols as i64 { continue; }
                            let p = pos_in_path[nr2 as usize][nc2 as usize];
                            let is_right = p > m;
                            let is_left  = p + 1 < l; // p < l-1
                            if !is_right && !is_left { continue; }

                            // 条件3: king_adj(path[l], path[p+1])
                            if is_right {
                                if p + 1 < n2 && !king_adj(path[l], path[p + 1]) { continue; }
                            } else {
                                if !king_adj(path[l], path[p + 1]) { continue; }
                            }

                            let sum_b  = psum[m + 1] - psum[l];
                            let wseg_b = wsum[m + 1] - wsum[l];
                            let delta = if is_right {
                                let sum_c = psum[p + 1] - psum[mp1];
                                let d = (p - m) as i64;
                                -(k as i64) * sum_c
                                    + (d - k as i64 + 1 + 2 * m as i64) * sum_b
                                    - 2 * wseg_b
                            } else {
                                let sum_f = psum[l] - psum[p + 1];
                                let d_l = (l - 1 - p) as i64;
                                k as i64 * sum_f
                                    + (-d_l - k as i64 + 1 + 2 * m as i64) * sum_b
                                    - 2 * wseg_b
                            };

                            let accept = delta >= 0 || { let r = delta as f64 / temp; r > -30.0 && rng.next_f64() < r.exp() };
                            if accept {
                                buf[..k].copy_from_slice(&path[l..=m]);
                                if is_right {
                                    let new_start = p - k + 1;
                                    for i in l..new_start {
                                        path[i] = path[i + k];
                                        a_val[i] = a_val[i + k];
                                        pos_in_path[path[i].0][path[i].1] = i;
                                    }
                                    for i in 0..k {
                                        path[new_start + i] = buf[k - 1 - i];
                                        a_val[new_start + i] = a[buf[k - 1 - i].0][buf[k - 1 - i].1];
                                        pos_in_path[path[new_start + i].0][path[new_start + i].1] = new_start + i;
                                    }
                                    for i in l..=p {
                                        psum[i + 1] = psum[i] + a_val[i];
                                        wsum[i + 1] = wsum[i] + i as i64 * a_val[i];
                                    }
                                } else {
                                    let d_sz = l - 1 - p;
                                    for i in (0..d_sz).rev() {
                                        path[p + 1 + k + i] = path[p + 1 + i];
                                        a_val[p + 1 + k + i] = a_val[p + 1 + i];
                                        pos_in_path[path[p + 1 + k + i].0][path[p + 1 + k + i].1] = p + 1 + k + i;
                                    }
                                    for i in 0..k {
                                        path[p + 1 + i] = buf[k - 1 - i];
                                        a_val[p + 1 + i] = a[buf[k - 1 - i].0][buf[k - 1 - i].1];
                                        pos_in_path[path[p + 1 + i].0][path[p + 1 + i].1] = p + 1 + i;
                                    }
                                    for i in p+1..=m {
                                        psum[i + 1] = psum[i] + a_val[i];
                                        wsum[i + 1] = wsum[i] + i as i64 * a_val[i];
                                    }
                                }
                                oropt_accepted += 1;
                                break 'orrev;
                            }
                        }
                    }
                }
            }
        } else {
            // ── or-opt: [l..=m] を q の後ろへ移動 ───────────────────────────
            // 条件1保証: path[l-1] の king 隣接から m+1 を探す
            let (pr, pc) = path[l - 1];
            'oropt: for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if dr == 0 && dc == 0 { continue; }
                    let nr = pr as i64 + dr;
                    let nc = pc as i64 + dc;
                    if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 { continue; }
                    let mp1 = pos_in_path[nr as usize][nc as usize];
                    if mp1 <= l || mp1 >= n2 { continue; }
                    let k = mp1 - l;
                    if k > max_k { continue; }
                    let m = mp1 - 1;

                    // 条件2保証: path[l] の king 隣接から q を探す
                    let (lr, lc) = path[l];
                    for dr2 in -1i64..=1 {
                        for dc2 in -1i64..=1 {
                            if dr2 == 0 && dc2 == 0 { continue; }
                            let nr2 = lr as i64 + dr2;
                            let nc2 = lc as i64 + dc2;
                            if nr2 < 0 || nr2 >= rows as i64 || nc2 < 0 || nc2 >= cols as i64 { continue; }
                            let q = pos_in_path[nr2 as usize][nc2 as usize];
                            let is_right = q > m && q - m <= oropt_max_dist;
                            let is_left  = q + 1 < l; // q < l-1
                            if !is_right && !is_left { continue; }

                            // 条件3: king_adj(path[m], path[q+1])
                            if is_right {
                                if q + 1 < n2 && !king_adj(path[m], path[q + 1]) { continue; }
                            } else {
                                // q+1 <= l-1 <= m, always valid index
                                if !king_adj(path[m], path[q + 1]) { continue; }
                            }

                            oropt_iters += 1;

                            let sum_seg1 = psum[m + 1] - psum[l];
                            let delta = if is_right {
                                let sum_inter = psum[q + 1] - psum[mp1];
                                (q - m) as i64 * sum_seg1 - k as i64 * sum_inter
                            } else {
                                let d = l - 1 - q;
                                let sum_inter = psum[l] - psum[q + 1];
                                k as i64 * sum_inter - d as i64 * sum_seg1
                            };

                            let accept = delta >= 0 || { let r = delta as f64 / temp; r > -30.0 && rng.next_f64() < r.exp() };
                            if accept {
                                buf[..k].copy_from_slice(&path[l..=m]);
                                if is_right {
                                    let new_start = q - k + 1;
                                    for i in l..new_start {
                                        path[i] = path[i + k];
                                        a_val[i] = a_val[i + k];
                                        pos_in_path[path[i].0][path[i].1] = i;
                                    }
                                    for i in 0..k {
                                        path[new_start + i] = buf[i];
                                        a_val[new_start + i] = a[buf[i].0][buf[i].1];
                                        pos_in_path[path[new_start + i].0][path[new_start + i].1] = new_start + i;
                                    }
                                    for i in l..=q {
                                        psum[i + 1] = psum[i] + a_val[i];
                                        wsum[i + 1] = wsum[i] + i as i64 * a_val[i];
                                    }
                                } else {
                                    let d = l - 1 - q;
                                    for i in (0..d).rev() {
                                        path[q + 1 + k + i] = path[q + 1 + i];
                                        a_val[q + 1 + k + i] = a_val[q + 1 + i];
                                        pos_in_path[path[q + 1 + k + i].0][path[q + 1 + k + i].1] = q + 1 + k + i;
                                    }
                                    for i in 0..k {
                                        path[q + 1 + i] = buf[i];
                                        a_val[q + 1 + i] = a[buf[i].0][buf[i].1];
                                        pos_in_path[path[q + 1 + i].0][path[q + 1 + i].1] = q + 1 + i;
                                    }
                                    for i in q+1..=m {
                                        psum[i + 1] = psum[i] + a_val[i];
                                        wsum[i + 1] = wsum[i] + i as i64 * a_val[i];
                                    }
                                }
                                oropt_accepted += 1;
                                break 'oropt;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("twoopt: iters={twoopt_iters} acc={twoopt_accepted}");
    eprintln!("oropt:  iters={oropt_iters} acc={oropt_accepted}");
}

#[inline(always)]
fn king_adj(a: Pos, b: Pos) -> bool {
    a.0.abs_diff(b.0).max(a.1.abs_diff(b.1)) == 1
}

fn validate_path(path: &[Pos], n: usize) {
    assert_eq!(path.len(), n * n);
    let mut used = vec![vec![false; n]; n];
    for &(r, c) in path { assert!(!used[r][c]); used[r][c] = true; }
    for i in 1..path.len() { assert!(king_adj(path[i-1], path[i]), "non-adjacent at {i}"); }
}

fn raw_score(a: &[Vec<i64>], path: &[Pos]) -> i64 {
    path.iter().enumerate().map(|(t, &(r, c))| t as i64 * a[r][c]).sum()
}

fn display_score(raw: i64, n2: i64) -> i64 { (raw + n2 / 2) / n2 }
