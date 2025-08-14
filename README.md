# evictor

[![Crates.io](https://img.shields.io/crates/v/evictor.svg)](https://crates.io/crates/evictor)
[![Docs.rs](https://docs.rs/evictor/badge.svg)](https://docs.rs/evictor)
[![Dependency status](https://deps.rs/repo/github/jesterhearts/evictor/status.svg)](https://deps.rs/repo/github/jesterhearts/evictor)

Provides several cache implementations with different eviction policies:

- **LRU (Least Recently Used)** - Evicts the item that was accessed longest ago
- **MRU (Most Recently Used)** - Evicts the item that was accessed most recently  
- **LFU (Least Frequently Used)** - Evicts the item that has been accessed least frequently

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

### Creating from Iterator

All cache types can be created from iterators. This will set the capacity to the number of items in the iterator:

```rust
use evictor::{Lru, Mru, Lfu};

let items = vec![
    (1, "one".to_string()),
    (2, "two".to_string()),
    (3, "three".to_string()),
];

// Works with any cache type
let lru_cache: Lru<i32, String> = items.clone().into_iter().collect();
let mru_cache: Mru<i32, String> = items.clone().into_iter().collect();
let lfu_cache: Lfu<i32, String> = items.into_iter().collect();
```

### Common Operations

All cache types support the same operations with identical APIs:

```rust
use std::num::NonZeroUsize;
use evictor::Lru; // Could also use Mru or Lfu

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

## License

This project is licensed under the either the APACHE or MIT License at your option. See the
[LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) files for details.