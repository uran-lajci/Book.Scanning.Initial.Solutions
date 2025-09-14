from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys, os, random

# ============================
# Data models (optimized with slots)
# ============================

@dataclass(frozen=True, slots=True)
class Library:
    id: int
    signup_days: int
    ship_per_day: int
    books: List[int]

@dataclass(frozen=True, slots=True)
class Instance:
    B: int
    L: int
    D: int
    book_scores: List[int]
    libraries: List[Library]

@dataclass(slots=True)
class Solution:
    library_order: List[int]
    chosen_books: Dict[int, List[int]]

# ============================
# Input parsing (fast text read + split)
# ============================

def parse_instance_from_path(path: str) -> Instance:
    with open(path, 'r', encoding='utf-8', newline='') as file:
        toks = file.read().split()
    it = iter(map(int, toks))

    B, L, D = next(it), next(it), next(it)
    book_scores = [next(it) for _ in range(B)]

    libraries: List[Library] = []
    for lib_id in range(L):
        N, T, M = next(it), next(it), next(it)
        books = [next(it) for _ in range(N)]
        libraries.append(Library(lib_id, T, M, books))

    return Instance(B, L, D, book_scores, libraries)

# ============================
# Scoring / evaluation (delta capable)
# ============================

def evaluate(instance: Instance, solution: Solution) -> int:
    """Full evaluation with tight inner loop (bytearray membership)."""
    B = instance.B
    D = instance.D
    libs = instance.libraries
    scores = instance.book_scores
    order = solution.library_order
    chosen = solution.chosen_books

    used = bytearray(B)
    total = 0
    days_spent = 0

    for lid in order:
        lib = libs[lid]
        days_spent += lib.signup_days
        days_left = D - days_spent
        if days_left <= 0:
            continue
        cap = days_left * lib.ship_per_day
        if cap <= 0:
            continue
        lst = chosen.get(lid)
        if not lst:
            continue
        u = used; s = scores; c = cap
        for b in lst:
            if c == 0:
                break
            if not u[b]:
                u[b] = 1
                total += s[b]
                c -= 1
    return total


def _eval_prefix_state(instance: Instance, solution: Solution, stop_idx: int) -> Tuple[bytearray, int, int]:
    """Return (used_flags, total_score, days_spent) after processing order[:stop_idx]."""
    B = instance.B
    D = instance.D
    libs = instance.libraries
    scores = instance.book_scores
    order = solution.library_order
    chosen = solution.chosen_books

    used = bytearray(B)
    total = 0
    days_spent = 0

    for i in range(stop_idx):
        lid = order[i]
        lib = libs[lid]
        days_spent += lib.signup_days
        days_left = D - days_spent
        if days_left <= 0:
            continue
        cap = days_left * lib.ship_per_day
        if cap <= 0:
            continue
        lst = chosen.get(lid)
        if not lst:
            continue
        u = used; s = scores; c = cap
        for b in lst:
            if c == 0:
                break
            if not u[b]:
                u[b] = 1; total += s[b]; c -= 1

    return used, total, days_spent


def evaluate_from(instance: Instance, solution: Solution, start_idx: int) -> int:
    """Evaluate candidate by reusing prefix up to start_idx (assuming prefix unchanged)."""
    used, total, days_spent = _eval_prefix_state(instance, solution, start_idx)

    D = instance.D
    libs = instance.libraries
    scores = instance.book_scores
    order = solution.library_order
    chosen = solution.chosen_books

    for i in range(start_idx, len(order)):
        lid = order[i]
        lib = libs[lid]
        days_spent += lib.signup_days
        days_left = D - days_spent
        if days_left <= 0:
            continue
        cap = days_left * lib.ship_per_day
        if cap <= 0:
            continue
        lst = chosen.get(lid)
        if not lst:
            continue
        u = used; s = scores; c = cap
        for b in lst:
            if c == 0:
                break
            if not u[b]:
                u[b] = 1; total += s[b]; c -= 1
    return total

# ============================
# Output writing (buffered)
# ============================

def write_solution_to_path(solution: Solution, path: str) -> None:
    order = [lid for lid in solution.library_order if solution.chosen_books.get(lid)]
    lines: List[str] = [f"{len(order)}\n"]
    for lid in order:
        books = solution.chosen_books[lid]
        lines.append(f"{lid} {len(books)}\n")
        lines.append((" ".join(map(str, books)) + "\n") if books else "\n")
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lines)

# ============================
# Helpers & precomputation
# ============================

def _build_pos_map(order: List[int]) -> Dict[int, int]:
    return {lid: i for i, lid in enumerate(order)}


def _catalog_sets(instance: Instance) -> List[set[int]]:
    """Precompute per-library catalog sets for fast membership checks."""
    return [set(lib.books) for lib in instance.libraries]


def repair_solution(instance: Instance, sol: Solution) -> Solution:
    """Return a structurally valid solution (subset-of-catalog, unique libs, no dup books per lib).
    Does NOT simulate time; purely structural repair for output validity.
    """
    catalogs = _catalog_sets(instance)

    # 1) dedup library order (keep first occurrences)
    seen_libs: set[int] = set()
    new_order: List[int] = []
    for lid in sol.library_order:
        if 0 <= lid < instance.L and lid not in seen_libs:
            seen_libs.add(lid)
            new_order.append(lid)

    # 2) filter chosen books to each library's catalog & remove duplicates preserving order
    new_chosen: Dict[int, List[int]] = {}
    for lid in new_order:
        lst = sol.chosen_books.get(lid, []) or []
        cat = catalogs[lid]
        seen_b: set[int] = set()
        fixed: List[int] = []
        for b in lst:
            if 0 <= b < instance.B and b in cat and b not in seen_b:
                seen_b.add(b)
                fixed.append(b)
        new_chosen[lid] = fixed

    return Solution(library_order=new_order, chosen_books=new_chosen)


def validate_solution(instance: Instance, sol: Solution) -> Tuple[bool, str]:
    catalogs = _catalog_sets(instance)
    # libs unique and in range
    if len(sol.library_order) != len(set(sol.library_order)):
        return False, "Duplicate libraries in order"
    for lid in sol.library_order:
        if not (0 <= lid < instance.L):
            return False, f"Library id out of range: {lid}"
        lst = sol.chosen_books.get(lid, [])
        seen = set()
        for b in lst:
            if not (0 <= b < instance.B):
                return False, f"Book id out of range: {b} in lib {lid}"
            if b not in catalogs[lid]:
                return False, f"Book {b} not in library {lid} catalog"
            if b in seen:
                return False, f"Duplicate book {b} in library {lid}"
            seen.add(b)
    return True, "ok"

def compute_prefix_signup_days(order: List[int], libs: List[Library]) -> List[int]:
    pref: List[int] = []
    acc = 0
    for lid in order:
        acc += libs[lid].signup_days
        pref.append(acc)
    return pref


def build_score_cache(instance: Instance) -> Tuple[List[List[int]], List[List[int]]]:
    scores = instance.book_scores
    sorted_books: List[List[int]] = []
    prefix_sums: List[List[int]] = []
    for lib in instance.libraries:
        srt = sorted(lib.books, key=lambda b: scores[b], reverse=True)
        sorted_books.append(srt)
        ps = [0]
        acc = 0
        for b in srt:
            acc += scores[b]
            ps.append(acc)
        prefix_sums.append(ps)
    return sorted_books, prefix_sums


def sum_top_k(prefix_sums: List[List[int]], lib_id: int, k: int) -> int:
    ps = prefix_sums[lib_id]
    k = max(0, min(k, len(ps) - 1))
    return ps[k]


def library_priority(instance: Instance, lib_id: int, sorted_books: List[List[int]], prefix_sums: List[List[int]]) -> float:
    lib = instance.libraries[lib_id]
    k = min(50, len(sorted_books[lib_id]))
    if k == 0:
        return 0.0
    avg_top = sum_top_k(prefix_sums, lib_id, k) / k
    return (lib.ship_per_day * avg_top) / (lib.signup_days + 1)


def _top_k_books_by_score(instance: Instance, lib: Library, k: int) -> List[int]:
    if k <= 0:
        return []
    k = min(k, len(lib.books))
    return sorted(lib.books, key=lambda b: instance.book_scores[b], reverse=True)[:k]


def _estimate_capacity(instance: Instance, lib: Library) -> int:
    return max(0, instance.D - lib.signup_days) * lib.ship_per_day


def clone_solution(sol: Solution) -> Solution:
    return Solution(library_order=sol.library_order[:],
                    chosen_books={lid: lst[:] for lid, lst in sol.chosen_books.items()})

# ============================
# Neighborhood moves (random) → return (candidate, start_idx)
# ============================

def move_swap_signed_with_signed(sol: Solution, i: int, j: int) -> Tuple[Solution, int]:
    if i == j or not (0 <= i < len(sol.library_order)) or not (0 <= j < len(sol.library_order)):
        return sol, len(sol.library_order)
    new = clone_solution(sol)
    new.library_order[i], new.library_order[j] = new.library_order[j], new.library_order[i]
    return new, min(i, j)


def move_swap_signed_with_unsigned(instance: Instance, sol: Solution, pos: int, new_lib_id: int) -> Tuple[Solution, int]:
    if not (0 <= pos < len(sol.library_order)):
        return sol, len(sol.library_order)
    if new_lib_id in sol.library_order:
        return sol, len(sol.library_order)
    new = clone_solution(sol)
    new.library_order[pos] = new_lib_id
    if new_lib_id not in new.chosen_books:
        lib = instance.libraries[new_lib_id]
        cap = _estimate_capacity(instance, lib)
        new.chosen_books[new_lib_id] = _top_k_books_by_score(instance, lib, cap)
    return new, pos


def move_swap_books_between_libs(instance: Instance, sol: Solution, lib_a: int, lib_b: int,
                                  pos_map: Dict[int, int],
                                  book_a: Optional[int] = None, book_b: Optional[int] = None) -> Tuple[Solution, int]:
    if lib_a == lib_b or lib_a not in pos_map or lib_b not in pos_map:
        return sol, len(sol.library_order)

    a_list = sol.chosen_books.get(lib_a, [])
    b_list = sol.chosen_books.get(lib_b, [])
    if not a_list or not b_list:
        return sol, len(sol.library_order)

    set_a = set(a_list)
    set_b = set(b_list)

    if book_a is None:
        book_a = next((b for b in a_list if b in set_b), a_list[0])
    if book_b is None:
        book_b = next((b for b in b_list if b in set_a), b_list[0])

    new = clone_solution(sol)
    a_new = new.chosen_books.get(lib_a, [])
    b_new = new.chosen_books.get(lib_b, [])

    if book_a in a_new:
        a_new.remove(book_a)
    if book_b in b_new:
        b_new.remove(book_b)
    if book_b not in a_new:
        a_new.append(book_b)
    if book_a not in b_new:
        b_new.append(book_a)

    new.chosen_books[lib_a] = a_new
    new.chosen_books[lib_b] = b_new

    start = min(pos_map[lib_a], pos_map[lib_b])
    return new, start


def move_swap_last_with_prior(sol: Solution, lib_id: int, pos_map: Dict[int, int]) -> Tuple[Solution, int]:
    lst = sol.chosen_books.get(lib_id)
    if not lst or len(lst) < 2 or lib_id not in pos_map:
        return sol, len(sol.library_order)
    new = clone_solution(sol)
    lst2 = new.chosen_books[lib_id]
    j = random.randrange(0, len(lst2) - 1)
    lst2[-1], lst2[j] = lst2[j], lst2[-1]
    return new, pos_map[lib_id]


def move_swap_neighbor_libraries(sol: Solution, idx: int) -> Tuple[Solution, int]:
    if not (0 <= idx < len(sol.library_order) - 1):
        return sol, len(sol.library_order)
    new = clone_solution(sol)
    new.library_order[idx], new.library_order[idx + 1] = new.library_order[idx + 1], new.library_order[idx]
    return new, idx


def move_insert_library(instance: Instance, sol: Solution, lib_id: int, pos: int) -> Tuple[Solution, int]:
    if lib_id in sol.library_order:
        return sol, len(sol.library_order)
    pos = max(0, min(pos, len(sol.library_order)))
    new = clone_solution(sol)
    new.library_order.insert(pos, lib_id)
    if lib_id not in new.chosen_books:
        lib = instance.libraries[lib_id]
        cap = _estimate_capacity(instance, lib)
        new.chosen_books[lib_id] = _top_k_books_by_score(instance, lib, cap)
    return new, pos

# ============================
# Guided operators (heuristic based)
# ============================

def guided_swap_neighbor_by_priority(instance: Instance, sol: Solution,
                                     sorted_books: List[List[int]], prefix_sums: List[List[int]]) -> Tuple[Solution, int]:
    n = len(sol.library_order)
    if n < 2:
        return sol, n
    idx = random.randrange(n - 1)
    a = sol.library_order[idx]
    b = sol.library_order[idx + 1]
    if library_priority(instance, b, sorted_books, prefix_sums) > library_priority(instance, a, sorted_books, prefix_sums):
        new = clone_solution(sol)
        new.library_order[idx], new.library_order[idx + 1] = new.library_order[idx + 1], new.library_order[idx]
        return new, idx
    return sol, n


def guided_promote_across_cutoff(instance: Instance, sol: Solution, pos_map: Dict[int, int]) -> Tuple[Solution, int]:
    if not sol.library_order:
        return sol, len(sol.library_order)
    lid = random.choice(sol.library_order)
    idx = pos_map[lid]
    pref = compute_prefix_signup_days(sol.library_order, instance.libraries)
    finish = pref[idx]
    days_left = instance.D - finish
    if days_left <= 0:
        return sol, len(sol.library_order)
    cap = days_left * instance.libraries[lid].ship_per_day
    lst = sol.chosen_books.get(lid, [])
    if not lst or cap <= 0 or len(lst) <= cap:
        return sol, len(sol.library_order)

    scores = instance.book_scores
    prefix_idx = min(range(cap), key=lambda i: scores[lst[i]])
    tail_idx = cap + max(range(len(lst) - cap), key=lambda j: scores[lst[cap + j]])
    if scores[lst[tail_idx]] <= scores[lst[prefix_idx]]:
        return sol, len(sol.library_order)

    new = clone_solution(sol)
    L = new.chosen_books[lid]
    L[prefix_idx], L[tail_idx] = L[tail_idx], L[prefix_idx]
    return new, idx


def guided_insert_best_unsigned(instance: Instance, sol: Solution,
                                sorted_books: List[List[int]], prefix_sums: List[List[int]]) -> Tuple[Solution, int]:
    unsigned = list(set(range(instance.L)) - set(sol.library_order))
    if not unsigned:
        return sol, len(sol.library_order)

    def optimistic_value(lib_id: int) -> int:
        lib = instance.libraries[lib_id]
        cap = max(0, instance.D - lib.signup_days) * lib.ship_per_day
        return sum_top_k(prefix_sums, lib_id, cap)

    unsigned.sort(key=optimistic_value, reverse=True)
    candidates = unsigned[: min(5, len(unsigned))]

    n = len(sol.library_order)
    positions = [0] if n == 0 else sorted({0, n // 3, (2 * n) // 3, n})

    best_sol = sol
    best_start = len(sol.library_order)
    best_score = -1
    base_score = evaluate(instance, sol)

    for lib_id in candidates:
        for pos in positions:
            cand, start_idx = move_insert_library(instance, sol, lib_id, pos)
            if start_idx >= len(cand.library_order):
                continue
            cand_score = evaluate_from(instance, cand, start_idx)
            if cand_score > best_score and cand_score >= base_score:
                best_sol, best_start, best_score = cand, start_idx, cand_score

    return (best_sol, best_start) if best_score >= base_score else (sol, len(sol.library_order))

# ============================
# Exhaustive-but-local operators
# ============================

# E1) Best signed-signed swap in a small window around a random center

def exhaustive_best_swap_in_window(instance: Instance, sol: Solution, window: int = 6) -> Tuple[Solution, int]:
    n = len(sol.library_order)
    if n < 2:
        return sol, n
    center = random.randrange(n)
    lo = max(0, center - window)
    hi = min(n - 1, center + window)

    best_sol = sol
    best_start = n
    best_score = evaluate(instance, sol)

    for i in range(lo, hi):
        for j in range(i + 1, hi + 1):
            cand, start_idx = move_swap_signed_with_signed(sol, i, j)
            cand_score = evaluate_from(instance, cand, start_idx)
            if cand_score > best_score:
                best_sol, best_start, best_score = cand, start_idx, cand_score
    return (best_sol, best_start) if best_sol is not sol else (sol, n)

# E2) Best relocation of one signed library within a small window

def exhaustive_best_relocate_in_window(instance: Instance, sol: Solution, window: int = 6) -> Tuple[Solution, int]:
    n = len(sol.library_order)
    if n < 2:
        return sol, n
    i = random.randrange(n)
    lo = max(0, i - window)
    hi = min(n, i + window + 1)

    base_score = evaluate(instance, sol)
    best_sol = sol
    best_start = n
    best_score = base_score

    for pos in range(lo, hi):
        if pos == i:
            continue
        # simulate relocate: remove at i, insert at pos
        cand = clone_solution(sol)
        lid = cand.library_order.pop(i)
        cand.library_order.insert(pos if pos < i else pos - 1, lid)
        start_idx = min(i, pos)
        cand_score = evaluate_from(instance, cand, start_idx)
        if cand_score > best_score:
            best_sol, best_start, best_score = cand, start_idx, cand_score
    return (best_sol, best_start) if best_sol is not sol else (sol, n)

# E3) Best insertion of a top-k unsigned library into a small set of positions

def exhaustive_best_insert_unsigned_window(instance: Instance, sol: Solution, k_libs: int = 3, window: int = 6) -> Tuple[Solution, int]:
    unsigned = list(set(range(instance.L)) - set(sol.library_order))
    if not unsigned:
        return sol, len(sol.library_order)

    # rank by optimistic bound
    scores_cache = build_score_cache(instance)[1]
    def optimistic_value(lib_id: int) -> int:
        lib = instance.libraries[lib_id]
        cap = max(0, instance.D - lib.signup_days) * lib.ship_per_day
        return sum_top_k(scores_cache, lib_id, cap)

    unsigned.sort(key=optimistic_value, reverse=True)
    cand_libs = unsigned[: min(k_libs, len(unsigned))]

    n = len(sol.library_order)
    if n == 0:
        positions = [0]
    else:
        center = random.randrange(n + 1)
        lo = max(0, center - window)
        hi = min(n, center + window)
        positions = list(range(lo, hi + 1))

    base_score = evaluate(instance, sol)
    best_sol = sol
    best_start = n
    best_score = base_score

    for lid in cand_libs:
        for pos in positions:
            cand, start_idx = move_insert_library(instance, sol, lid, pos)
            if start_idx >= len(cand.library_order):
                continue
            cand_score = evaluate_from(instance, cand, start_idx)
            if cand_score > best_score:
                best_sol, best_start, best_score = cand, start_idx, cand_score
    return (best_sol, best_start) if best_sol is not sol else (sol, n)

# E4) Best across-cutoff book swap using small prefix/tail windows

def exhaustive_best_cutoff_swap(instance: Instance, sol: Solution, pos_map: Dict[int, int], prefix_range: int = 5, tail_range: int = 5) -> Tuple[Solution, int]:
    if not sol.library_order:
        return sol, len(sol.library_order)
    lid = random.choice(sol.library_order)
    idx = pos_map[lid]
    pref = compute_prefix_signup_days(sol.library_order, instance.libraries)
    finish = pref[idx]
    days_left = instance.D - finish
    if days_left <= 0:
        return sol, len(sol.library_order)
    cap = days_left * instance.libraries[lid].ship_per_day
    lst = sol.chosen_books.get(lid, [])
    if not lst or cap <= 0 or len(lst) <= cap:
        return sol, len(sol.library_order)

    scores = instance.book_scores
    P = list(range(max(0, cap - prefix_range), cap))
    T = list(range(cap, min(len(lst), cap + tail_range)))
    if not P or not T:
        return sol, len(sol.library_order)

    base_score = evaluate(instance, sol)
    best_sol = sol
    best_start = len(sol.library_order)
    best_score = base_score

    for i in P:
        for j in T:
            cand = clone_solution(sol)
            L = cand.chosen_books[lid]
            L[i], L[j] = L[j], L[i]
            cand_score = evaluate_from(instance, cand, idx)
            if cand_score > best_score:
                best_sol, best_start, best_score = cand, idx, cand_score
    return (best_sol, best_start) if best_sol is not sol else (sol, len(sol.library_order))

# ============================
# Random initial solution (capacity-aware)
# ============================

def generate_random_solution(instance: Instance, cap_per_lib: int = 0, seed: int | None = None) -> Solution:
    if seed is not None:
        random.seed(seed)

    lib_ids = list(range(instance.L))
    random.shuffle(lib_ids)

    chosen_books: Dict[int, List[int]] = {}
    for lib in instance.libraries:
        if not lib.books:
            continue
        cap = _estimate_capacity(instance, lib)
        if cap_per_lib > 0:
            cap = min(cap, cap_per_lib)
        k = min(cap, len(lib.books))
        if k <= 0:
            continue
        chosen = random.sample(lib.books, k)
        chosen_books[lib.id] = chosen

    return Solution(library_order=lib_ids, chosen_books=chosen_books)

# ============================
# Local Search mixing random + guided + exhaustive (delta evaluation)
# ============================

def local_search(instance: Instance, start: Solution, iterations: int = 7000, seed: Optional[int] = None,
                 guided_prob: float = 0.35, exhaustive_prob: float = 0.15) -> Tuple[Solution, int]:
    if seed is not None:
        random.seed(seed)

    best = clone_solution(start)
    best_score = evaluate(instance, best)

    current = clone_solution(best)
    current_score = best_score
    pos_map = _build_pos_map(current.library_order)

    sorted_books, prefix_sums = build_score_cache(instance)
    all_lib_ids = set(range(instance.L))

    for _ in range(iterations):
        r = random.random()
        candidate: Solution
        start_idx: int

        if r < exhaustive_prob:
            which = random.randrange(4)
            if which == 0:
                candidate, start_idx = exhaustive_best_swap_in_window(instance, current, window=6)
            elif which == 1:
                candidate, start_idx = exhaustive_best_relocate_in_window(instance, current, window=6)
            elif which == 2:
                candidate, start_idx = exhaustive_best_insert_unsigned_window(instance, current, k_libs=3, window=6)
            else:
                candidate, start_idx = exhaustive_best_cutoff_swap(instance, current, pos_map, prefix_range=5, tail_range=5)
        elif r < exhaustive_prob + guided_prob:
            which = random.randrange(3)
            if which == 0 and len(current.library_order) >= 2:
                candidate, start_idx = guided_swap_neighbor_by_priority(instance, current, sorted_books, prefix_sums)
            elif which == 1:
                candidate, start_idx = guided_promote_across_cutoff(instance, current, pos_map)
            else:
                candidate, start_idx = guided_insert_best_unsigned(instance, current, sorted_books, prefix_sums)
        else:
            move_type = random.randrange(1, 7)
            if move_type == 1 and len(current.library_order) >= 2:
                i, j = random.sample(range(len(current.library_order)), 2)
                candidate, start_idx = move_swap_signed_with_signed(current, i, j)
            elif move_type == 2 and len(current.library_order) >= 1:
                pos = random.randrange(len(current.library_order))
                unsigned = list(all_lib_ids - set(current.library_order))
                if not unsigned:
                    continue
                new_lib = random.choice(unsigned)
                candidate, start_idx = move_swap_signed_with_unsigned(instance, current, pos, new_lib)
            elif move_type == 3 and len(current.library_order) >= 2:
                a, b = random.sample(current.library_order, 2)
                candidate, start_idx = move_swap_books_between_libs(instance, current, a, b, pos_map)
            elif move_type == 4 and current.library_order:
                lid = random.choice(current.library_order)
                candidate, start_idx = move_swap_last_with_prior(current, lid, pos_map)
            elif move_type == 5 and len(current.library_order) >= 2:
                idx = random.randrange(len(current.library_order) - 1)
                candidate, start_idx = move_swap_neighbor_libraries(current, idx)
            elif move_type == 6:
                unsigned = list(all_lib_ids - set(current.library_order))
                if not unsigned:
                    continue
                lid = random.choice(unsigned)
                pos = random.randrange(len(current.library_order) + 1)
                candidate, start_idx = move_insert_library(instance, current, lid, pos)
            else:
                continue

        if start_idx >= len(current.library_order):
            continue

        cand_score = evaluate_from(instance, candidate, start_idx)

        if cand_score >= current_score:
            prev_order = current.library_order
            current, current_score = candidate, cand_score
            if current.library_order != prev_order:
                pos_map = _build_pos_map(current.library_order)
            if cand_score > best_score:
                best, best_score = candidate, cand_score

    return best, best_score

# ============================
# CLI (instance name -> ./instances/<name>.txt)
# ============================

def write_solution_to_path(solution: Solution, path: str) -> None:
    order = [lid for lid in solution.library_order if solution.chosen_books.get(lid)]
    lines: List[str] = [f"{len(order)}\n"]
    for lid in order:
        books = solution.chosen_books[lid]
        lines.append(f"{lid} {len(books)}\n")
        lines.append((" ".join(map(str, books)) + "\n") if books else "\n")
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lines)


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python script.py <instance_name>')
        sys.exit(1)

    INSTANCE_FOLDER = 'input'  # constant folder for instances
    instance_name = sys.argv[1]

    in_path = os.path.join(INSTANCE_FOLDER, instance_name + '.txt')
    out_path = os.path.join(INSTANCE_FOLDER, instance_name + '_out.txt')

    if not os.path.exists(in_path):
        print(f'Input not found: {in_path}')
        sys.exit(2)

    instance = parse_instance_from_path(in_path)

    # Random initial solution → Local Search with random + guided + exhaustive operators
    start = generate_random_solution(instance, cap_per_lib=0)
    best, score = local_search(instance, start, iterations=7000, seed=None, guided_prob=0.35, exhaustive_prob=0.15)

    write_solution_to_path(best, out_path)
    print(f'Score: {score} | Wrote solution to {out_path}')

if __name__ == '__main__':
    main()
