# evictor

[![Crates.io](https://img.shields.io/crates/v/evictor.svg)](https://crates.io/crates/evictor)
[![Docs.rs](https://docs.rs/evictor/badge.svg)](https://docs.rs/evictor)
[![Dependency status](https://deps.rs/repo/github/jesterhearts/evictor/status.svg)](https://deps.rs/repo/github/jesterhearts/evictor)

Provides several cache implementation defined by eviction policies, such as LRU (Least Recently
Used) and MRU (Most Recently Used).

## Usage
### Basic Example

```rust
use std::num::NonZeroUsize;

use evictor::Lru;

let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());

// Insert entries
cache.insert(1, "one".to_string());
cache.insert(2, "two".to_string());
cache.insert(3, "three".to_string());

// Access entry (marks as recently used)
cache.get(&1);

// Insert when full evicts least recently used
cache.insert(4, "four".to_string());
assert!(!cache.contains_key(&2)); // 2 was evicted

// Peek doesn't affect LRU order
cache.peek(&3);
let (oldest_key, _) = cache.tail().unwrap();
assert_eq!(oldest_key, &3);
```

### Creating from Iterator

This will set the capacity to the number of items in the iterator.

```rust
use evictor::Lru;

let items = vec![
    (1, "one".to_string()),
    (2, "two".to_string()),
    (3, "three".to_string()),
];

let cache: Lru<i32, String> = items.into_iter().collect();
```

## Features

### Default Features

- `ahash` - Fast hashing using the `ahash` crate (enabled by default)

## License

This project is licensed under the either the APACHE or MIT License at your option. See the
[LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) files for details.