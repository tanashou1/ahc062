#!/usr/bin/env bash
# eval.sh — Build solver, run on 10 test cases, score each, save results
# Usage: ./eval.sh [num_cases]   (default: 10)

set -euo pipefail

NUM=${1:-10}
TOOLS_DIR="tools/tools"
IN_DIR="$TOOLS_DIR/in"
OUT_DIR="results"
SCORE_LOG="$OUT_DIR/history.log"

# ── Build ─────────────────────────────────────────────────────────────────────
echo "[1/3] Building solver..."
source ~/.cargo/env
cargo build --release -q 2>&1

echo "[2/3] Building tools (vis)..."
cargo build --release -q --manifest-path "$TOOLS_DIR/Cargo.toml" 2>&1

mkdir -p "$OUT_DIR"

# ── Run & Score ───────────────────────────────────────────────────────────────
echo "[3/3] Running $NUM test cases..."
total=0
scores=()

for i in $(seq 0 $((NUM - 1))); do
    id=$(printf "%04d" $i)
    in_file="$IN_DIR/$id.txt"
    out_file="$OUT_DIR/$id.txt"

    if [[ ! -f "$in_file" ]]; then
        echo "  [$id] input not found, skipping"
        continue
    fi

    # Run solver (capture stderr for debug info)
    ./target/release/ahc062 < "$in_file" > "$out_file" 2>/tmp/solver_stderr_$id.txt
    dbg=$(cat /tmp/solver_stderr_$id.txt)

    # Score via vis (outputs "Score = N")
    score_line=$("$TOOLS_DIR/target/release/vis" "$in_file" "$out_file" 2>/dev/null)
    score=$(echo "$score_line" | grep -oP '(?<=Score = )\d+')

    total=$((total + score))
    scores+=("$id:$score")
    echo "  [$id] score=$score  ($dbg)"
done

avg=$((total / NUM))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total ($NUM cases): $total"
echo "  Average:            $avg"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Save history ──────────────────────────────────────────────────────────────
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")

{
    echo "# $timestamp  commit=$git_hash  cases=$NUM"
    for s in "${scores[@]}"; do echo "  $s"; done
    echo "  total=$total  avg=$avg"
    echo ""
} >> "$SCORE_LOG"

echo ""
echo "Results saved to $OUT_DIR/  |  History: $SCORE_LOG"
