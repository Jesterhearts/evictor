# evictor

[![Crates.io](https://img.shields.io/crates/v/evictor.svg)](https://crates.io/crates/evictor)
[![Docs.rs](https://docs.rs/evictor/badge.svg)](https://docs.rs/evictor)
[![Dependency status](https://deps.rs/repo/github/jesterhearts/evictor/status.svg)](https://deps.rs/repo/github/jesterhearts/evictor)

Provides several cache implementations with different eviction policies:

- **LRU (Least Recently Used)** - Evicts the item that was accessed longest ago
- **MRU (Most Recently Used)** - Evicts the item that was accessed most recently  
- **LFU (Least Frequently Used)** - Evicts the item that has been accessed least frequently
- **FIFO (First In, First Out)** - Evicts the item that was inserted earliest, regardless of access
  patterns
- **LIFO (Last In, First Out)** - Evicts the item that was inserted most recently, regardless of
  access patterns
- **Random** - Evicts a randomly selected item when the cache is full
- **SIEVE** - Efficient eviction using visited bits and hand pointer. See the
  [paper](https://junchengyang.com/publication/nsdi24-SIEVE.pdf) for details.

All caches are generic over key and value types, with a configurable capacity.

## Usage

### LRU (Least Recently Used) Cache

Evicts the item that was accessed longest ago:

```rust
use std::num::NonZeroUsize;
use evictor::Lru;

let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());

cache.insert(1, "one".to_string());
cache.insert(2, "two".to_string());
cache.insert(3, "three".to_string());

// Access entry (marks as recently used)
cache.get(&1);

// Insert when full evicts least recently used
cache.insert(4, "four".to_string());
assert!(!cache.contains_key(&2)); // 2 was evicted (least recently used)
assert!(cache.contains_key(&1));  // 1 was recently accessed, so kept
```

### MRU (Most Recently Used) Cache

Evicts the item that was accessed most recently:

```rust
use std::num::NonZeroUsize;
use evictor::Mru;

let mut cache = Mru::<i32, String>::new(NonZeroUsize::new(3).unwrap());

cache.insert(1, "one".to_string());
cache.insert(2, "two".to_string());
cache.insert(3, "three".to_string());

// Access entry (marks as recently used)
cache.get(&1);

// Insert when full evicts most recently used
cache.insert(4, "four".to_string());
assert!(!cache.contains_key(&1)); // 1 was evicted (most recently used)
assert!(cache.contains_key(&2));  // 2 was not recently accessed, so kept
```

### LFU (Least Frequently Used) Cache

Evicts the item that has been accessed least frequently:

```rust
use std::num::NonZeroUsize;
use evictor::Lfu;

let mut cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());

cache.insert(1, "one".to_string());
cache.insert(2, "two".to_string());
cache.insert(3, "three".to_string());

// Access entries different numbers of times
cache.get(&1); // frequency: 1
cache.get(&1); // frequency: 2
cache.get(&2); // frequency: 1
// Key 3 has frequency: 0 (never accessed after insertion)

// Insert when full evicts least frequently used
cache.insert(4, "four".to_string());
assert!(!cache.contains_key(&3)); // 3 was evicted (frequency 0)
assert!(cache.contains_key(&1));  // 1 has highest frequency (2)
assert!(cache.contains_key(&2));  // 2 has frequency 1
```

### FIFO (First In, First Out) Cache

Evicts the item that was inserted earliest, regardless of access patterns:

```rust
use std::num::NonZeroUsize;
use evictor::FIFO;

let mut cache = FIFO::<i32, String>::new(NonZeroUsize::new(3).unwrap());

cache.insert(1, "one".to_string());
cache.insert(2, "two".to_string());
cache.insert(3, "three".to_string());

// Access entry (doesn't affect eviction order in FIFO)
cache.get(&1);

// Insert when full evicts first inserted
cache.insert(4, "four".to_string());
assert!(!cache.contains_key(&1)); // 1 was evicted (first inserted)
assert!(cache.contains_key(&2));  // 2 was inserted second, so kept
assert!(cache.contains_key(&3));  // 3 was inserted third, so kept
```

### LIFO (Last In, First Out) Cache

Evicts the item that was inserted most recently, regardless of access patterns:

```rust
use std::num::NonZeroUsize;
use evictor::LIFO;

let mut cache = LIFO::<i32, String>::new(NonZeroUsize::new(3).unwrap());

cache.insert(1, "one".to_string());
cache.insert(2, "two".to_string());
cache.insert(3, "three".to_string());

// Access entry (doesn't affect eviction order in LIFO)
cache.get(&1);

// Insert when full evicts most recently inserted
cache.insert(4, "four".to_string());
assert!(!cache.contains_key(&3)); // 3 was evicted (most recently inserted)
assert!(cache.contains_key(&1));  // 1 was inserted first, so kept
assert!(cache.contains_key(&2));  // 2 was inserted second, so kept
```

### Random Cache

Evicts a randomly selected item when the cache is full. Useful as a baseline for comparison with
other policies or when no particular access pattern can be predicted:

```rust
use std::num::NonZeroUsize;
use evictor::Random;

let mut cache = Random::<i32, String>::new(NonZeroUsize::new(3).unwrap());

cache.insert(1, "one".to_string());
cache.insert(2, "two".to_string());
cache.insert(3, "three".to_string());

// Access entry (doesn't affect eviction order in Random policy)
cache.get(&1);

// Insert when full evicts a random item
cache.insert(4, "four".to_string());
// One of the original entries (1, 2, or 3) was randomly evicted
assert_eq!(cache.len(), 3);
assert!(cache.contains_key(&4)); // 4 was just inserted
```

### SIEVE Cache

Efficient eviction using visited bits and hand pointer. SIEVE provides performance comparable to
LRU. It gives recently accessed items a "second chance" before eviction:

```rust
use std::num::NonZeroUsize;
use evictor::SIEVE;

let mut cache = SIEVE::<i32, String>::new(NonZeroUsize::new(3).unwrap());

cache.insert(1, "one".to_string());
cache.insert(2, "two".to_string());
cache.insert(3, "three".to_string());

// Access entry to set visited bit (gives it a second chance)
cache.get(&1);

// Insert when full - SIEVE scans for unvisited entries to evict
cache.insert(4, "four".to_string());
assert!(cache.contains_key(&1));  // 1 gets second chance (was accessed)
assert!(!cache.contains_key(&2)); // 2 was evicted (not accessed)
assert!(cache.contains_key(&3));  // 3 remains
assert!(cache.contains_key(&4));  // 4 was just inserted
```

### Creating from Iterator

All cache types can be created from iterators. This will set the capacity to the number of items in
the iterator:

```rust
use evictor::{Lru, Mru, Lfu, FIFO, LIFO, Random, SIEVE};

let items = vec![
    (1, "one".to_string()),
    (2, "two".to_string()),
    (3, "three".to_string()),
];

// Works with any cache type
let lru_cache: Lru<i32, String> = items.clone().into_iter().collect();
let mru_cache: Mru<i32, String> = items.clone().into_iter().collect();
let lfu_cache: Lfu<i32, String> = items.clone().into_iter().collect();
let fifo_cache: FIFO<i32, String> = items.clone().into_iter().collect();
let lifo_cache: LIFO<i32, String> = items.clone().into_iter().collect();
let random_cache: Random<i32, String> = items.clone().into_iter().collect();
let sieve_cache: SIEVE<i32, String> = items.into_iter().collect();
```

### Common Operations

All cache types support the same operations with identical APIs:

```rust
use std::num::NonZeroUsize;
use evictor::Lru; // Could also use Mru, Lfu, FIFO, LIFO, Random, or SIEVE

let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());

// Insert and access
cache.insert(1, "value".to_string());
cache.insert(2, "another".to_string());
let value = cache.get(&1); // Returns Some(&String) and updates cache order

// Non-mutating operations (don't affect eviction order)
let value = cache.peek(&1); // Returns Some(&String) without updating order
let exists = cache.contains_key(&1); // Returns bool
if let Some((key, value)) = cache.tail() {
    println!("Next to be evicted: {} -> {}", key, value);
}

// Other operations
cache.remove(&1); // Remove specific key
if let Some((key, value)) = cache.pop() {
    println!("Removed: {} -> {}", key, value);
}
cache.clear(); // Remove all entries
```

## Features

### Default Features

- `ahash` - Fast hashing using the `ahash` crate (enabled by default)
- `rand` - Enables the Random cache policy using the `rand` crate.

## License

This project is licensed under the either the APACHE or MIT License at your option. See the
[LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) files for details.