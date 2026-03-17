use std::collections::HashSet;
use std::io::{self, Read};

const AREA_SIZE: usize = 10;
const BASE_G0_PATH: [Pos; 50] = [
    (4, 0), (3, 1), (2, 0), (1, 0), (0, 0), (0, 1), (1, 1), (0, 2), (1, 3), (2, 2),
    (3, 3), (2, 4), (3, 5), (4, 4), (5, 3), (4, 2), (5, 1), (6, 0), (7, 1), (6, 2),
    (7, 3), (8, 2), (9, 3), (8, 4), (9, 5), (8, 6), (7, 5), (6, 4), (5, 5), (4, 6),
    (5, 7), (6, 6), (7, 7), (8, 8), (9, 7), (9, 8), (9, 9), (8, 9), (7, 9), (6, 8),
    (5, 9), (4, 8), (3, 9), (2, 8), (3, 7), (2, 6), (1, 7), (0, 6), (1, 5), (0, 4),
];
const BASE_G1_DOWN_PATH: [Pos; 50] = [
    (0, 3), (1, 2), (2, 1), (3, 0), (4, 1), (5, 0), (6, 1), (5, 2), (6, 3), (7, 2),
    (8, 1), (7, 0), (8, 0), (9, 0), (9, 1), (9, 2), (8, 3), (9, 4), (8, 5), (7, 4),
    (6, 5), (7, 6), (6, 7), (5, 6), (4, 7), (3, 6), (4, 5), (5, 4), (4, 3), (3, 2),
    (2, 3), (3, 4), (2, 5), (1, 4), (0, 5), (1, 6), (2, 7), (1, 8), (0, 7), (0, 8),
    (0, 9), (1, 9), (2, 9), (3, 8), (4, 9), (5, 8), (6, 9), (7, 8), (8, 7), (9, 6),
];
const BASE_G1_RIGHT_PATH: [Pos; 50] = [
    (0, 3), (1, 2), (2, 1), (3, 0), (4, 1), (5, 0), (6, 1), (5, 2), (4, 3), (3, 2),
    (2, 3), (3, 4), (2, 5), (1, 4), (0, 5), (1, 6), (0, 7), (0, 8), (1, 9), (2, 9),
    (3, 8), (4, 9), (5, 8), (6, 9), (7, 8), (6, 7), (7, 6), (8, 7), (9, 6), (8, 5),
    (9, 4), (8, 3), (9, 2), (9, 1), (9, 0), (8, 0), (7, 0), (8, 1), (7, 2), (6, 3),
    (7, 4), (6, 5), (5, 4), (4, 5), (5, 6), (4, 7), (3, 6), (2, 7), (1, 8), (0, 9),
];

type Pos = (usize, usize);

#[derive(Clone)]
struct Candidate {
    path: Vec<Pos>,
    start: Pos,
    end: Pos,
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut iter = input.split_ascii_whitespace();

    let n: usize = iter.next().unwrap().parse().unwrap();
    let mut a = vec![vec![0i64; n]; n];
    for row in &mut a {
        for x in row {
            *x = iter.next().unwrap().parse().unwrap();
        }
    }

    let path = solve(&a, n);
    validate_path(&path, n);

    let raw = raw_score(&a, &path);
    eprintln!("score={}", display_score(raw, (n * n) as i64));

    for (r, c) in path {
        println!("{r} {c}");
    }
}

fn solve(a: &[Vec<i64>], n: usize) -> Vec<Pos> {
    assert_eq!(n % AREA_SIZE, 0);
    let area_n = n / AREA_SIZE;
    let candidates = generate_candidates();
    let area_orders = generate_area_orders(area_n);

    let mut best_raw = i64::MIN;
    let mut best_path = Vec::new();

    for order in area_orders {
        let path = build_path_for_order(a, &order, &candidates);
        let raw = raw_score(a, &path);
        if raw > best_raw {
            best_raw = raw;
            best_path = path;
        }
    }

    best_path
}

fn build_path_for_order(a: &[Vec<i64>], order: &[(usize, usize)], candidates: &[Candidate]) -> Vec<Pos> {
    let cand_n = candidates.len();
    let area_count = order.len();
    let mut local_scores = vec![vec![0i64; cand_n]; area_count];
    let mut starts = vec![vec![(0usize, 0usize); cand_n]; area_count];
    let mut ends = vec![vec![(0usize, 0usize); cand_n]; area_count];

    for (idx, &(ar, ac)) in order.iter().enumerate() {
        let base_r = ar * AREA_SIZE;
        let base_c = ac * AREA_SIZE;
        let offset = idx * AREA_SIZE * AREA_SIZE;
        for (ci, cand) in candidates.iter().enumerate() {
            let mut score = 0i64;
            for (dt, &(r, c)) in cand.path.iter().enumerate() {
                score += (offset + dt) as i64 * a[base_r + r][base_c + c];
            }
            local_scores[idx][ci] = score;
            starts[idx][ci] = (base_r + cand.start.0, base_c + cand.start.1);
            ends[idx][ci] = (base_r + cand.end.0, base_c + cand.end.1);
        }
    }

    let mut parent = vec![vec![usize::MAX; cand_n]; area_count];
    let mut dp_prev = local_scores[0].clone();

    for idx in 1..area_count {
        let mut dp_cur = vec![i64::MIN; cand_n];
        for cur in 0..cand_n {
            let cur_start = starts[idx][cur];
            let local = local_scores[idx][cur];
            for prev in 0..cand_n {
                if dp_prev[prev] == i64::MIN || !king_adj(ends[idx - 1][prev], cur_start) {
                    continue;
                }
                let cand_score = dp_prev[prev] + local;
                if cand_score > dp_cur[cur] {
                    dp_cur[cur] = cand_score;
                    parent[idx][cur] = prev;
                }
            }
        }
        dp_prev = dp_cur;
    }

    let mut last = 0usize;
    for i in 1..cand_n {
        if dp_prev[i] > dp_prev[last] {
            last = i;
        }
    }
    assert!(dp_prev[last] != i64::MIN, "failed to connect area templates");

    let mut chosen = vec![0usize; area_count];
    let mut cur = last;
    for idx in (0..area_count).rev() {
        chosen[idx] = cur;
        if idx > 0 {
            cur = parent[idx][cur];
            assert!(cur != usize::MAX, "broken parent chain");
        }
    }

    let mut path = Vec::with_capacity(a.len() * a.len());
    for (idx, &(ar, ac)) in order.iter().enumerate() {
        let base_r = ar * AREA_SIZE;
        let base_c = ac * AREA_SIZE;
        for &(r, c) in &candidates[chosen[idx]].path {
            path.push((base_r + r, base_c + c));
        }
    }
    path
}

fn generate_candidates() -> Vec<Candidate> {
    let mut bases = Vec::new();
    let mut turn = Vec::with_capacity(AREA_SIZE * AREA_SIZE);
    turn.extend_from_slice(&BASE_G0_PATH);
    turn.extend_from_slice(&BASE_G1_DOWN_PATH);
    bases.push(turn);

    let mut straight = Vec::with_capacity(AREA_SIZE * AREA_SIZE);
    straight.extend_from_slice(&BASE_G0_PATH);
    straight.extend_from_slice(&BASE_G1_RIGHT_PATH);
    bases.push(straight);

    let mut seen: HashSet<Vec<Pos>> = HashSet::new();
    let mut out = Vec::new();

    for base in bases {
        for rev in [false, true] {
            let seq: Vec<Pos> = if rev {
                base.iter().rev().copied().collect()
            } else {
                base.clone()
            };
            for sym in 0..8 {
                let transformed: Vec<Pos> = seq
                    .iter()
                    .map(|&p| transform_in_square(p, AREA_SIZE, sym))
                    .collect();
                if seen.insert(transformed.clone()) {
                    validate_local_template(&transformed);
                    out.push(Candidate {
                        start: transformed[0],
                        end: transformed[transformed.len() - 1],
                        path: transformed,
                    });
                }
            }
        }
    }

    out
}

fn validate_local_template(path: &[Pos]) {
    assert_eq!(path.len(), AREA_SIZE * AREA_SIZE);
    let mut used = [[false; AREA_SIZE]; AREA_SIZE];
    for &p in path {
        assert!(!used[p.0][p.1], "duplicate cell in local template");
        used[p.0][p.1] = true;
    }
    for i in 1..path.len() {
        assert!(king_adj(path[i - 1], path[i]), "broken local template edge");
    }

    let first_group = local_group(path[0]);
    let second_group = local_group(path[AREA_SIZE * AREA_SIZE / 2]);
    assert_ne!(first_group, second_group, "template must switch groups exactly once");
    for &p in &path[..AREA_SIZE * AREA_SIZE / 2] {
        assert_eq!(local_group(p), first_group, "first half mixes groups");
    }
    for &p in &path[AREA_SIZE * AREA_SIZE / 2..] {
        assert_eq!(local_group(p), second_group, "second half mixes groups");
    }
}

fn generate_area_orders(area_n: usize) -> Vec<Vec<Pos>> {
    let mut base = Vec::with_capacity(area_n * area_n);
    for r in 0..area_n {
        if r % 2 == 0 {
            for c in 0..area_n {
                base.push((r, c));
            }
        } else {
            for c in (0..area_n).rev() {
                base.push((r, c));
            }
        }
    }

    let mut out = Vec::new();
    let mut seen: HashSet<Vec<Pos>> = HashSet::new();
    for sym in 0..8 {
        let order: Vec<Pos> = base
            .iter()
            .map(|&p| transform_in_square(p, area_n, sym))
            .collect();
        if seen.insert(order.clone()) {
            out.push(order);
        }
    }
    out
}

fn transform_in_square(mut p: Pos, size: usize, sym: usize) -> Pos {
    let last = size - 1;
    let rot = sym / 2;
    let mirror = sym % 2 == 1;
    for _ in 0..rot {
        p = (p.1, last - p.0);
    }
    if mirror {
        p.1 = last - p.1;
    }
    p
}

fn local_group((r, c): Pos) -> usize {
    match (r, c) {
        (0, 1) | (1, 0) | (8, 9) | (9, 8) => 0,
        (0, 8) | (1, 9) | (8, 0) | (9, 1) => 1,
        _ => (r + c) & 1,
    }
}

fn validate_path(path: &[Pos], n: usize) {
    assert_eq!(path.len(), n * n, "wrong path length");
    let mut used = vec![vec![false; n]; n];
    for &(r, c) in path {
        assert!(r < n && c < n, "path goes out of board");
        assert!(!used[r][c], "duplicate cell in full path");
        used[r][c] = true;
    }
    for i in 1..path.len() {
        assert!(king_adj(path[i - 1], path[i]), "non-adjacent full path edge");
    }
}

fn king_adj(a: Pos, b: Pos) -> bool {
    a.0.abs_diff(b.0).max(a.1.abs_diff(b.1)) == 1
}

fn raw_score(a: &[Vec<i64>], path: &[Pos]) -> i64 {
    path.iter()
        .enumerate()
        .map(|(t, &(r, c))| t as i64 * a[r][c])
        .sum()
}

fn display_score(raw: i64, n2: i64) -> i64 {
    (raw + n2 / 2) / n2
}
