use std::io::{self, BufWriter, Read, Write};
use std::time::Instant;

fn main() {
    let start = Instant::now();
    let input = {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).unwrap();
        buf
    };
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let mut scanner = Scanner::new(&input);
    let result = solve(&mut scanner, start);

    writeln!(out, "{}", result).unwrap();
}

fn solve(sc: &mut Scanner, start: Instant) -> String {
    // TODO: read input according to the problem statement
    let _n: usize = sc.next();

    let time_limit = std::time::Duration::from_millis(1900);

    // Main loop: improve solution within time limit
    let answer = String::new();
    while start.elapsed() < time_limit {
        // TODO: implement solution
        break;
    }

    answer
}

// --------------- Fast scanner ---------------

pub struct Scanner<'a> {
    iter: std::str::SplitAsciiWhitespace<'a>,
}

impl<'a> Scanner<'a> {
    pub fn new(s: &'a str) -> Self {
        Self {
            iter: s.split_ascii_whitespace(),
        }
    }

    pub fn next<T: std::str::FromStr>(&mut self) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.iter.next().unwrap().parse().unwrap()
    }

    pub fn next_vec<T: std::str::FromStr>(&mut self, n: usize) -> Vec<T>
    where
        T::Err: std::fmt::Debug,
    {
        (0..n).map(|_| self.next()).collect()
    }
}
