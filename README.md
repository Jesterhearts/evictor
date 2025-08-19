# evictor

[![Crates.io](https://img.shields.io/crates/v/evictor.svg)](https://crates.io/crates/evictor)
[![Docs.rs](https://docs.rs/evictor/badge.svg)](https://docs.rs/evictor)
[![Dependency status](https://deps.rs/repo/github/jesterhearts/evictor/status.svg)](https://deps.rs/repo/github/jesterhearts/evictor)

Generic in‑memory caches with pluggable eviction policies. One uniform API, generic over key + value, O(1) average core ops, and interchangeable eviction behavior chosen by a type alias.

## Policies at a glance

| Policy | Evicts                       | Primary signal                | Extra per‑entry bytes (approx)               | Notes                                         |
| ------ | ---------------------------- | ----------------------------- | -------------------------------------------- | --------------------------------------------- |
| Lru    | Least recently used          | Recency                       | 2 * usize (prev/next)                        | Doubly linked list of entries                 |
| Mru    | Most recently used           | Recency (inverted)            | 2 * usize                                    | Same structure as LRU, opposite victim        |
| Lfu    | Least frequently used        | Frequency + recency tie break | 2 * usize + u64 + (bucket index bookkeeping) | Frequency buckets; O(1) amortized updates     |
| Fifo   | Oldest inserted              | Insertion order               | 2 * usize                                    | Queue semantics (head victim)                 |
| Lifo   | Newest inserted              | Insertion order               | 2 * usize                                    | Stack semantics (head victim)                 |
| Random | Random entry                 | RNG                           | None                                         | Requires `rand` feature (default)             |
| Sieve  | First unvisited (2nd chance) | Visited bit pass              | 2 * usize + 1 byte                           | Implements NSDI'24 SIEVE (see [paper][paper]) |


## Quick start

### LRU example (API identical for all policies)

```rust
use std::num::NonZeroUsize;
use evictor::Lru;

let mut cache = Lru::<i32, &'static str>::new(NonZeroUsize::new(3).unwrap());
cache.insert(1, "one");
cache.insert(2, "two");
cache.insert(3, "three");
cache.get(&1);           // mark as recently used
cache.insert(4, "four"); // evicts key 2
assert!(!cache.contains_key(&2));
```

Swap `Lru` for `Mru`, `Lfu`, `Fifo`, `Lifo`, `Random`, or `Sieve` to change behavior.

### Policy nuances

```rust
use std::num::NonZeroUsize;
use evictor::{Mru, Lfu, Fifo, Lifo, Random, Sieve};

let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());
mru.insert('a', 1);
mru.insert('b', 2);
mru.get(&'a');         // 'a' now MRU
mru.insert('c', 3);    // evicts 'a'

let mut lfu = Lfu::new(NonZeroUsize::new(2).unwrap());
lfu.insert('a', 1);
lfu.insert('b', 2);
lfu.get(&'a');         // freq(a)=1
lfu.insert('c', 3);    // evicts 'b'

let mut sieve = Sieve::new(NonZeroUsize::new(2).unwrap());
sieve.insert(1, "x");
sieve.insert(2, "y");
sieve.get(&1);         // mark visited
sieve.insert(3, "z");  // evicts 2 (unvisited)
```

### Building from iterators

`FromIterator` collects all pairs setting capacity to the number of unique items in the iterator
(minimum 1).

```rust
use evictor::{Lru, Mru, Lfu, Fifo, Lifo, Random, Sieve};

let data = [(1, "one".to_string()), (2, "two".to_string()), (3, "three".to_string())];
let lru: Lru<_, _> = data.into_iter().collect();
assert_eq!(lru.len(), 3);
```

### Iteration order

`iter()` and `into_iter()` yield entries in eviction order (tail first) except `Random` (arbitrary /
randomized).

### Performance

Average `insert`, `get`, `peek`, `remove`, `pop`: O(1). Worst cases inherit `IndexMap` behavior. LFU
adjustments are O(1) amortized (bucket promotion); SIEVE's `tail()` may scan O(n) in degenerate
patterns.

### Mutation‑aware access (`peek_mut`)

`peek_mut` only reorders if the value is mutated (tracked by Drop). Plain `get_mut` always touches.

### Capacity management

* `set_capacity(new)` adjusts capacity, evicting as needed.
* `evict_to_size(n)` trims down to a target size.
* `evict_n_entries(k)` evicts up to `k` victims.

### Feature flags

| Feature              | Default | Purpose                                                      |
| -------------------- | ------- | ------------------------------------------------------------ |
| `ahash`              | Yes     | Faster hashing (`ahash::RandomState`)                        |
| `rand`               | Yes     | Enables `Random` policy                                      |
| `statistics`         | No      | Enables cache statistics tracking (hits, misses, etc.)       |
| `internal-debugging` | No      | Potentially expensive invariant checking (debug builds only) |

### Safety & Guarantees

* No unsafe code in library (although e.g. indexmap may use some).

### License

MIT OR Apache‑2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

[paper]: https://junchengyang.com/publication/nsdi24-SIEVE.pdf