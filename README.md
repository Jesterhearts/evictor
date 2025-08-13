# evictor

A fixed-capacity least-recently-used (LRU) cache implementation in Rust.

## Usage
### Basic Example

```rust
use evictor::Lru;

let mut cache = Lru::<3, i32, String>::new();

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
let (oldest_key, _) = cache.oldest().unwrap();
assert_eq!(oldest_key, &3);
```

### Creating from Iterator

```rust
use evictor::Lru;

let items = vec![
    (1, "one".to_string()),
    (2, "two".to_string()),
    (3, "three".to_string()),
];

let cache: Lru<5, i32, String> = items.into_iter().collect();
```

## Features

### Default Features

- `ahash` - Fast hashing using the `ahash` crate (enabled by default)

### Disabling ahash

To use `std::hash::RandomState` instead:

```toml
[dependencies]
evictor = { version = "0.1.0", default-features = false }
```

## License

This project is licensed under the either the APACHE or MIT License at your option. See the [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) files for details.