#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]
#![cfg_attr(all(doc, ENABLE_DOC_AUTO_CFG), feature(doc_auto_cfg))]
#![forbid(unsafe_code)]

mod lfu;
mod lru;
mod mru;
mod queue;
#[cfg(feature = "rand")]
mod random;
mod r#ref;
mod sieve;
mod utils;

use std::{
    hash::Hash,
    num::NonZeroUsize,
};

use indexmap::IndexMap;
pub use lfu::LfuPolicy;
pub use lru::LruPolicy;
pub use mru::MruPolicy;
pub use queue::{
    FifoPolicy,
    LifoPolicy,
};
#[cfg(feature = "rand")]
pub use random::RandomPolicy;
pub use r#ref::Entry;
pub use sieve::SievePolicy;

#[cfg(not(feature = "ahash"))]
type RandomState = std::hash::RandomState;
#[cfg(feature = "ahash")]
type RandomState = ahash::RandomState;

mod private {
    pub trait Sealed {}
}

/// Trait for wrapping values in cache entries.
///
/// This trait allows policies to associate additional metadata with cached
/// values while providing uniform access to the underlying data. Different
/// eviction policies may store additional information alongside the cached
/// value (such as access frequency for Lfu or access timestamps).
///
/// # Type Parameters
///
/// * `T` - The type of the cached value being wrapped
///
/// # Design Notes
///
/// This trait abstracts over the storage requirements of different eviction
/// policies. Some policies like Lru only need to store the value itself,
/// while others like Lfu need to track additional metadata such as access
/// frequency counts. By using this trait, the cache implementation can
/// remain generic over different storage strategies.
#[doc(hidden)]
pub trait EntryValue<T>: private::Sealed {
    /// Creates a new entry containing the given value.
    ///
    /// This method should initialize any policy-specific metadata to
    /// appropriate default values. For example, an Lfu policy might
    /// initialize access counts to zero, while an Lru policy might not need
    /// any additional initialization.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to be stored in the cache entry
    fn new(value: T) -> Self;

    /// Converts the entry into its contained value.
    ///
    /// This method consumes the entry and extracts the wrapped value,
    /// discarding any policy-specific metadata. This is typically used
    /// when an entry is being removed from the cache.
    fn into_value(self) -> T;

    /// Returns a reference to the contained value.
    ///
    /// This method provides read-only access to the cached value without
    /// affecting any policy-specific metadata or consuming the entry.
    fn value(&self) -> &T;

    /// Returns a mutable reference to the contained value.
    ///
    /// This method provides write access to the cached value. The policy
    /// may or may not update its metadata when this method is called,
    /// depending on the specific implementation requirements.
    fn value_mut(&mut self) -> &mut T;
}

/// Trait for cache metadata that tracks eviction state.
///
/// This trait provides policy-specific metadata storage and the ability
/// to identify which entry should be evicted next. Different eviction
/// policies maintain different types of metadata to efficiently determine
/// eviction order.
///
/// # Design Philosophy
///
/// The metadata abstraction allows each eviction policy to maintain its
/// own internal state without exposing implementation details to the
/// generic cache structure. This enables efficient O(1) eviction decisions
/// across all supported policies.
///
/// # Metadata Examples by Policy
///
/// - **Lru/Mru**: Typically maintains a doubly-linked list order via indices
/// - **Lfu**: May track frequency buckets and least-used indices
/// - **Fifo/Lifo**: Often just tracks head/tail pointers for queue ordering
/// - **Random**: May maintain no additional state or seed information
///
/// # Default Implementation
///
/// The `Default` trait bound ensures that metadata can be easily initialized
/// when creating a new cache. The default state should represent an empty
/// cache with no entries.
#[doc(hidden)]
pub trait Metadata<T>: Default + private::Sealed {
    /// The entry type used by this policy to wrap cached values.
    type EntryType: EntryValue<T>;

    /// Returns the index of the entry that should be evicted next.
    ///
    /// This method provides the core eviction logic by identifying which
    /// entry in the cache should be removed when space is needed. The
    /// returned index corresponds to a position in the cache's internal
    /// storage (typically an `IndexMap`).
    ///
    /// # Returns
    ///
    /// The index of the entry that would be evicted next according to
    /// this policy's eviction strategy. For an empty cache, this method's
    /// behavior is unspecified.
    ///
    /// # Performance
    ///
    /// This method must have O(1) time complexity to maintain the cache's
    /// performance guarantees. Implementations should pre-compute or
    /// efficiently track the tail index rather than scanning entries.
    ///
    /// # Policy-Specific Behavior
    ///
    /// - **Lru**: Returns index of least recently used entry
    /// - **Mru**: Returns index of most recently used entry
    /// - **Lfu**: Returns index of least frequently used entry
    /// - **Fifo**: Returns index of first inserted entry
    /// - **Lifo**: Returns index of last inserted entry
    /// - **Random**: Returns index of randomly selected entry
    ///
    /// # Usage
    ///
    /// This method is typically called by the cache implementation when:
    /// - The cache is at capacity and a new entry needs to be inserted
    /// - The `pop()` method is called to explicitly remove the tail entry
    /// - The `tail()` method is called to inspect the next eviction candidate
    fn candidate_removal_index<K>(
        &self,
        queue: &IndexMap<K, Self::EntryType, RandomState>,
    ) -> usize;
}

/// Trait defining cache eviction policies.
///
/// This trait encapsulates the behavior of different cache eviction strategies
/// such as Lru, Mru, Lfu, Fifo, Lifo, and Random. Each policy defines how
/// entries are prioritized and which entry should be evicted when the cache is
/// full.
///
/// # Type Parameters
///
/// * `T` - The type of values stored in the cache
///
/// # Associated Types
///
/// This trait defines three associated types that work together:
/// - `EntryType`: Wraps cached values with policy-specific metadata
/// - `MetadataType`: Maintains global policy state for eviction decisions
/// - `IntoIter`: Provides owned iteration over cache contents in eviction order
///
/// # Policy Implementations
///
/// The crate provides several built-in policy implementations:
///
/// ## Recency-Based Policies
/// - **Lru (Least Recently Used)**: Evicts entries that haven't been accessed
///   recently
/// - **Mru (Most Recently Used)**: Evicts entries that were accessed most
///   recently
///
/// ## Frequency-Based Policies  
/// - **Lfu (Least Frequently Used)**: Evicts entries with the lowest access
///   count
///
/// ## Insertion-Order Policies
/// - **Fifo (First In, First Out)**: Evicts entries in insertion order
/// - **Lifo (Last In, First Out)**: Evicts entries in reverse insertion order
///
/// ## Randomized Policies
/// - **Random**: Evicts entries randomly
///
/// ## Sieve Policy
/// - **Sieve**: Uses a visited bit and hand pointer for efficient eviction with
///   second-chance semantics
///
/// # Error Handling
///
/// Policy implementations should handle edge cases gracefully:
/// - Empty caches (no entries to evict)
/// - Single-entry caches
/// - Invalid indices (though these should not occur in normal operation)
///
/// # Iteration Order Consistency
///
/// The `iter()` and `into_iter()` methods must return entries in eviction
/// order, meaning the first item returned should be the same as what `tail()`
/// would return (except for policies where the order is specified as
/// arbitrary e.g. [`Random`]).
#[doc(hidden)]
pub trait Policy<T>: private::Sealed {
    /// The metadata type used by this policy to track eviction state.
    type MetadataType: Metadata<T>;

    /// The iterator type returned by `into_iter()`.
    type IntoIter<K>: Iterator<Item = (K, T)>;

    /// Updates the entry's position in the eviction order.
    ///
    /// This method is called when an entry is accessed (via get, insert, etc.)
    /// and should update the policy's internal state to reflect the access.
    ///
    /// # Parameters
    /// - `index`: The index of the accessed entry
    /// - `metadata`: Mutable reference to policy metadata
    /// - `queue`: Mutable reference to the cache's storage
    ///
    /// # Returns
    /// The new index of the entry after any reordering
    fn touch_entry<K>(
        index: usize,
        make_room: bool,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<K, <Self::MetadataType as Metadata<T>>::EntryType, RandomState>,
    ) -> usize;

    /// Evict the next entry from the cache. Returns the actual evicted index.
    #[allow(clippy::type_complexity)]
    fn evict_entry<K>(
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<K, <Self::MetadataType as Metadata<T>>::EntryType, RandomState>,
    ) -> (
        usize,
        Option<(K, <Self::MetadataType as Metadata<T>>::EntryType)>,
    ) {
        let removed = metadata.candidate_removal_index(queue);
        (removed, Self::swap_remove_entry(removed, metadata, queue))
    }

    /// Removes the entry at the specified index and returns the index of the
    /// entry which replaced it.
    ///
    /// This method handles the removal of an entry and updates the policy's
    /// internal state to maintain consistency.
    ///
    /// **Note**: After the removal of the item at `index`, the element at
    /// `index` **must be** the last element in the queue. That is, if you have
    /// a queue like `[0, 1, 2, 3]`, and you remove index 1, the queue must
    /// become `[0, 3, 2]`. This is relied on for insertion and retain
    /// operations.
    ///
    /// # Parameters
    /// - `index`: The index of the entry to remove
    /// - `metadata`: Mutable reference to policy metadata
    /// - `queue`: Mutable reference to the cache's storage
    ///
    /// # Returns
    /// - The removed key-value pair, or None if the index was invalid
    fn swap_remove_entry<K>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<K, <Self::MetadataType as Metadata<T>>::EntryType, RandomState>,
    ) -> Option<(K, <Self::MetadataType as Metadata<T>>::EntryType)>;

    /// Returns an iterator over the entries in the cache in eviction order.
    ///
    /// The iterator yields key-value pairs in the order they would be evicted,
    /// starting with the entry that would be evicted first (the "tail" of the
    /// eviction order). The specific ordering depends on the policy:
    ///
    /// - **Lru**: Iterates from least recently used to most recently used
    /// - **Mru**: Iterates from most recently used to least recently used
    /// - **Lfu**: Iterates from least frequently used to most frequently used,
    ///   with ties broken by insertion/access order
    /// - **Fifo**: Iterates from first inserted to last inserted
    /// - **Lifo**: Iterates from last inserted to first inserted
    /// - **Random**: Iterates in random order (debug) or arbitrary order
    ///   (release)
    ///
    /// This is an internal trait method used by policy implementations.
    ///
    /// # Parameters
    /// - `metadata`: Reference to policy-specific metadata for traversal
    /// - `queue`: Reference to the cache's underlying storage
    ///
    /// # Returns
    /// An iterator yielding `(&Key, &Value)` pairs in eviction order
    fn iter<'q, K>(
        metadata: &'q Self::MetadataType,
        queue: &'q IndexMap<K, <Self::MetadataType as Metadata<T>>::EntryType, RandomState>,
    ) -> impl Iterator<Item = (&'q K, &'q T)>
    where
        T: 'q;

    /// Converts the cache into an iterator over key-value pairs in eviction
    /// order.
    ///
    /// This method consumes the cache and returns an iterator that yields `(K,
    /// T)` pairs. The iteration order depends on the specific eviction
    /// policy:
    ///
    /// - **Lru**: Items are yielded from least recently used to most recently
    ///   used
    /// - **Mru**: Items are yielded from most recently used to least recently
    ///   used
    /// - **Lfu**: Items are yielded from least frequently used to most
    ///   frequently used
    /// - **Fifo**: Items are yielded from first inserted to last inserted
    /// - **Lifo**: Items are yielded from last inserted to first inserted
    /// - **Random**: Items are yielded in an arbitrary order.
    ///
    /// # Parameters
    /// - `metadata`: The policy's metadata containing eviction state
    ///   information
    /// - `queue`: The cache's internal storage map containing all key-value
    ///   pairs
    ///
    /// # Returns
    /// An iterator that yields `(K, T)` pairs in the policy-specific eviction
    /// order. The iterator takes ownership of the cache contents, making
    /// this a consuming operation.
    fn into_iter<K>(
        metadata: Self::MetadataType,
        queue: IndexMap<K, <Self::MetadataType as Metadata<T>>::EntryType, RandomState>,
    ) -> Self::IntoIter<K>;

    /// Converts the cache entries into an iterator of key-entry pairs in
    /// eviction order.
    ///
    /// This method is similar to `into_iter()` but returns the full entry
    /// wrappers instead of just the values. This allows access to any
    /// policy-specific metadata stored within the entries, which can be
    /// useful for debugging, analysis, or advanced cache operations.
    ///
    /// # Parameters
    ///
    /// * `metadata` - The policy's metadata containing eviction state
    ///   information
    /// * `queue` - The cache's internal storage map containing all key-entry
    ///   pairs
    ///
    /// # Returns
    ///
    /// An iterator that yields `(K, <Self::MetadataType as
    /// Metadata<T>>::EntryType)` pairs in the policy-specific eviction
    /// order. The iterator consumes the cache contents.
    ///
    /// # Use Cases
    ///
    /// This method is primarily used for internal implementation details where
    /// entry metadata is needed
    ///
    /// For typical usage, prefer `into_iter()` which extracts just the values.
    fn into_entries<K>(
        metadata: Self::MetadataType,
        queue: IndexMap<K, <Self::MetadataType as Metadata<T>>::EntryType, RandomState>,
    ) -> impl Iterator<Item = (K, <Self::MetadataType as Metadata<T>>::EntryType)>;

    /// Validates the cache's internal state against the full set of kv pairs
    /// known. This is **expensive** and should only be used for debugging
    /// purposes.
    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    #[doc(hidden)]
    fn debug_validate<K: Hash + Eq + std::fmt::Debug>(
        metadata: &Self::MetadataType,
        queue: &IndexMap<K, <Self::MetadataType as Metadata<T>>::EntryType, RandomState>,
    ) where
        T: std::fmt::Debug;
}

/// A least-recently-used (Lru) cache.
///
/// Evicts the entry that was accessed longest ago when the cache is full.
/// This policy is ideal for workloads where recently accessed items are
/// more likely to be accessed again.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(1) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop/Clear: O(1)
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Lru;
///
/// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// cache.get(&1); // Mark as recently used
/// cache.insert(4, "four".to_string()); // Evicts key 2
///
/// // `into_iter` returns the items in eviction order, which is Lru first.
/// assert_eq!(
///     cache.into_iter().collect::<Vec<_>>(),
///     [
///         (3, "three".to_string()),
///         (1, "one".to_string()),
///         (4, "four".to_string()),
///     ]
/// );
/// ```
#[doc(alias = "LRU")]
pub type Lru<Key, Value> = Cache<Key, Value, LruPolicy>;

/// See [`Lru`].
#[doc(hidden)]
pub type LRU<Key, Value> = Lru<Key, Value>;

/// A most-recently-used (Mru) cache.
///
/// Evicts the entry that was accessed most recently when the cache is full.
/// This policy can be useful for workloads with sequential access patterns
/// where the most recently accessed item is unlikely to be needed again.
///
/// # Time Complexity  
/// - Insert/Get/Remove: O(1) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop/Clear: O(1)
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Mru;
///
/// let mut cache = Mru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// cache.get(&1); // Mark as recently used
/// cache.insert(4, "four".to_string()); // Evicts key 1
///
/// // `into_iter` returns the items in eviction order, which is Mru first.
/// assert_eq!(
///     cache.into_iter().collect::<Vec<_>>(),
///     [
///         (4, "four".to_string()),
///         (3, "three".to_string()),
///         (2, "two".to_string()),
///     ]
/// );
/// ```
#[doc(alias = "MRU")]
pub type Mru<Key, Value> = Cache<Key, Value, MruPolicy>;

/// See [`Mru`].
#[doc(hidden)]
pub type MRU<Key, Value> = Mru<Key, Value>;

/// A least-frequently-used (Lfu) cache.
///
/// This implementation uses frequency buckets with doubly-linked lists to
/// achieve O(1) time complexity for all operations.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(1) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop/Clear: O(1)
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Lfu;
///
/// let mut cache = Lfu::<i32, String>::new(NonZeroUsize::new(4).unwrap());
///
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
/// cache.insert(4, "four".to_string());
///
/// // Access key 1 multiple times to increase its frequency
/// cache.get(&1);
/// cache.get(&1);
/// cache.get(&1);
///
/// // Access key 2, 3 once
/// cache.get(&2);
/// cache.get(&3);
///
/// // Key 4 has frequency 0 (never accessed after insertion)
/// // Insert new item - this will evict the least frequently used (key 4)
/// cache.insert(5, "five".to_string());
///
/// // `into_iter` returns the items in eviction order, which is Lfu first, with Lru tiebreaking.
/// assert_eq!(
///     cache.into_iter().collect::<Vec<_>>(),
///     [
///         (5, "five".to_string()),
///         (2, "two".to_string()),
///         (3, "three".to_string()),
///         (1, "one".to_string()),
///     ]
/// );
/// ```
#[doc(alias = "LFU")]
pub type Lfu<Key, Value> = Cache<Key, Value, LfuPolicy>;

/// See [`Lfu`].
#[doc(hidden)]
pub type LFU<Key, Value> = Lfu<Key, Value>;

/// A first-in-first-out (Fifo) cache.
///
/// Evicts the entry that was inserted earliest when the cache is full.
/// This policy treats the cache as a queue where the oldest inserted
/// item is removed first, regardless of access patterns.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(1) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop/Clear: O(1)
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Fifo;
///
/// let mut cache = Fifo::<i32, String>::new(NonZeroUsize::new(3).unwrap());
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// cache.get(&1); // Access key 1 - this doesn't affect eviction order in Fifo
/// cache.insert(4, "four".to_string()); // Evicts key 1 (first inserted)
///
/// // `into_iter` returns the items in eviction order, which is Fifo.
/// assert_eq!(
///     cache.into_iter().collect::<Vec<_>>(),
///     [
///         (2, "two".to_string()),
///         (3, "three".to_string()),
///         (4, "four".to_string()),
///     ]
/// );
/// ```
#[doc(alias = "FIFO")]
pub type Fifo<Key, Value> = Cache<Key, Value, FifoPolicy>;

/// See [`Fifo`].
#[doc(hidden)]
pub type FIFO<Key, Value> = Fifo<Key, Value>;

/// A last-in-first-out (Lifo) cache.
///
/// Evicts the entry that was inserted most recently when the cache is full.
/// This policy treats the cache as a stack where the most recently inserted
/// item is removed first, regardless of access patterns.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(1) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop/Clear: O(1)
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Lifo;
///
/// let mut cache = Lifo::<i32, String>::new(NonZeroUsize::new(3).unwrap());
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// cache.get(&1); // Access key 1 - this doesn't affect eviction order in Lifo
/// cache.insert(4, "four".to_string()); // Evicts key 3 (most recently inserted)
///
/// // `into_iter` returns the items in eviction order, which is Lifo.
/// assert_eq!(
///     cache.into_iter().collect::<Vec<_>>(),
///     [
///         (4, "four".to_string()),
///         (2, "two".to_string()),
///         (1, "one".to_string()),
///     ]
/// );
/// ```
#[doc(alias = "LIFO")]
pub type Lifo<Key, Value> = Cache<Key, Value, LifoPolicy>;

/// See [`Lifo`].
#[doc(hidden)]
pub type LIFO<Key, Value> = Lifo<Key, Value>;

/// A random eviction cache.
///
/// Evicts entries randomly when the cache is full. This policy can be useful
/// for workloads where no particular access pattern can be predicted, or as
/// a simple baseline for comparison with other eviction policies.
///
/// The random selection is performed using the `rand` crate. You **must
/// not** rely on the order returned by iteration. This is enforced by
/// randomizing the order of iteration during debug builds. Release builds do
/// not perform this randomization for performance reasons, but the order is
/// still arbitrary and should not be relied upon.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(1) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop/Clear: O(1)
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Random;
///
/// let mut cache = Random::<i32, String>::new(NonZeroUsize::new(3).unwrap());
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// // Access doesn't affect eviction order in Random policy
/// cache.get(&1);
/// cache.insert(4, "four".to_string()); // Evicts a random entry
///
/// // The order returned by `into_iter` is not predictable
/// let pairs: Vec<_> = cache.into_iter().collect();
/// assert_eq!(pairs.len(), 3);
/// ```
#[cfg(feature = "rand")]
pub type Random<Key, Value> = Cache<Key, Value, RandomPolicy>;

/// A Sieve cache implementing the algorithm outlined in the paper
/// [Sieve is Simpler than Lru](https://junchengyang.com/publication/nsdi24-SIEVE.pdf).
///
/// Sieve uses a "visited" bit per entry and a hand pointer that scans
/// for eviction candidates, giving recently accessed items a "second chance"
/// before eviction.
///
/// # Algorithm Overview
///
/// Sieve maintains:
/// - A **visited bit** for each cache entry (initially false)
/// - A **hand pointer** that moves through entries looking for eviction
///   candidates
/// - A **doubly-linked list** structure for efficient traversal
///
/// ## Key Operations
///
/// ### Access (get/get_mut)
/// When an entry is accessed:
/// 1. The visited bit is set to `true`
/// 2. The entry position in eviction order may be updated
///
/// ### Eviction
/// When space is needed:
/// 1. The hand pointer scans entries starting from its current position
/// 2. For each entry with `visited = true`: reset to `false` and continue
///    scanning
/// 3. The first entry with `visited = false` is evicted
/// 4. The hand pointer advances to the next position
///
/// This gives recently accessed entries a "second chance" - they're skipped
/// once during eviction scanning, but evicted if accessed again without being
/// used.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(1) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop/Clear: O(1) average, O(n) worst case
///
/// # Examples
///
/// ## Basic Usage
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::Sieve;
///
/// let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
/// cache.insert(1, "one");
/// cache.insert(2, "two");
/// cache.insert(3, "three");
///
/// // Access entry 1 to set its visited bit
/// cache.get(&1);
///
/// // This will evict entry 2 (unvisited), not entry 1
/// cache.insert(4, "four");
/// assert!(cache.contains_key(&1)); // Still present (second chance)
/// assert!(!cache.contains_key(&2)); // Evicted
/// ```
///
/// ## Second Chance Behavior
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::Sieve;
///
/// let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
/// cache.insert("A", 1);
/// cache.insert("B", 2);
///
/// // Access A to mark it as visited
/// cache.get(&"A");
///
/// // Insert C - will scan and find A visited, reset its bit, continue to B
/// cache.insert("C", 3);
/// assert!(cache.contains_key(&"A")); // A got second chance
/// assert!(!cache.contains_key(&"B")); // B was evicted
///
/// // Now A's visited bit is false, so next eviction will remove A
/// cache.insert("D", 4);
/// assert!(!cache.contains_key(&"A")); // A evicted (used its second chance)
/// assert!(cache.contains_key(&"C"));
/// assert!(cache.contains_key(&"D"));
/// ```
///
/// ## Comparison with peek() vs get()
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::Sieve;
///
/// let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
/// cache.insert(1, "one");
/// cache.insert(2, "two");
///
/// // peek() does NOT set visited bit
/// cache.peek(&1);
/// cache.insert(3, "three"); // Evicts 1 (not visited)
/// assert!(!cache.contains_key(&1));
///
/// // get() DOES set visited bit
/// cache.get(&2);
/// cache.insert(4, "four"); // Evicts 3 (2 gets second chance)
/// assert!(cache.contains_key(&2));
/// assert!(!cache.contains_key(&3));
/// ```
///
/// ## Iteration Order
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::Sieve;
///
/// let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
/// cache.insert("A", 1);
/// cache.insert("B", 2);
/// cache.insert("C", 3);
///
/// // Iteration follows eviction order (starting from hand position)
/// let items: Vec<_> = cache.iter().collect();
/// // Order depends on hand position and visited bits
/// assert_eq!(items.len(), 3);
///
/// // First item should match tail() (next eviction candidate)
/// assert_eq!(cache.tail().unwrap(), cache.iter().next().unwrap());
/// ```
#[doc(alias = "SIEVE")]
pub type Sieve<Key, Value> = Cache<Key, Value, SievePolicy>;

/// See [`Sieve`].
#[doc(hidden)]
pub type SIEVE<Key, Value> = Sieve<Key, Value>;

/// Cache performance and usage statistics.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "statistics")] {
/// use evictor::Lru;
///
/// let mut cache = Lru::new(std::num::NonZeroUsize::new(2).unwrap());
/// cache.insert("a".to_string(), 1);
/// cache.insert("b".to_string(), 2);
///
/// let stats = cache.statistics();
/// assert_eq!(stats.len(), 2);
/// assert_eq!(stats.insertions(), 2);
/// assert_eq!(stats.evictions(), 0);
/// # }
/// ```
#[derive(Clone, Debug)]
#[cfg(feature = "statistics")]
pub struct Statistics {
    len: usize,
    capacity: NonZeroUsize,
    evictions: u64,
    insertions: u64,
    misses: u64,
    hits: u64,
}

#[cfg(feature = "statistics")]
impl Statistics {
    fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            len: 0,
            capacity,
            evictions: 0,
            insertions: 0,
            misses: 0,
            hits: 0,
        }
    }

    /// Returns `true` if the cache contains no entries.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(2).unwrap());
    /// assert!(cache.statistics().is_empty());
    ///
    /// cache.insert("key".to_string(), "value".to_string());
    /// assert!(!cache.statistics().is_empty());
    /// # }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the current number of entries stored in the cache.
    ///
    /// This count represents the actual number of key-value pairs currently
    /// residing in the cache, which may be less than the cache's capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(3).unwrap());
    /// assert_eq!(cache.statistics().len(), 0);
    ///
    /// cache.insert("a".to_string(), 1);
    /// cache.insert("b".to_string(), 2);
    /// assert_eq!(cache.statistics().len(), 2);
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the current cache residency as a percentage (0.0 to 100.0).
    ///
    /// Residency represents how full the cache is, calculated as the current
    /// number of entries divided by the maximum capacity, expressed as a
    /// percentage. A residency of 100.0% indicates the cache is at full
    /// capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(4).unwrap());
    /// assert_eq!(cache.statistics().residency(), 0.0);
    ///
    /// cache.insert("a".to_string(), 1);
    /// cache.insert("b".to_string(), 2);
    /// assert_eq!(cache.statistics().residency(), 50.0);
    ///
    /// cache.insert("c".to_string(), 3);
    /// cache.insert("d".to_string(), 4);
    /// assert_eq!(cache.statistics().residency(), 100.0);
    /// # }
    /// ```
    pub fn residency(&self) -> f64 {
        self.len as f64 / self.capacity.get() as f64 * 100.0
    }

    /// Returns the maximum capacity of the cache.
    ///
    /// This is the maximum number of key-value pairs that the cache can store
    /// before eviction policies take effect to make room for new entries.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let cache: Lru<String, i32> = Lru::new(std::num::NonZeroUsize::new(100).unwrap());
    /// assert_eq!(cache.statistics().capacity(), 100);
    /// # }
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity.get()
    }

    /// Returns the total number of evictions that have occurred.
    ///
    /// An eviction happens when an entry is removed from the cache to make room
    /// for a new entry, according to the cache's eviction policy (LRU, LFU,
    /// etc.). This counter tracks the lifetime total of such removals.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(2).unwrap());
    /// assert_eq!(cache.statistics().evictions(), 0);
    ///
    /// cache.insert("a".to_string(), 1);
    /// cache.insert("b".to_string(), 2);
    /// assert_eq!(cache.statistics().evictions(), 0);
    ///
    /// // This insertion will evict the least recently used entry
    /// cache.insert("c".to_string(), 3);
    /// assert_eq!(cache.statistics().evictions(), 1);
    /// # }
    /// ```
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Returns the total number of insertion operations that have occurred.
    ///
    /// This counter increments every time a new key-value pair is inserted into
    /// the cache, regardless of whether it causes an eviction. Updates to
    /// existing keys also count as insertions.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(2).unwrap());
    /// assert_eq!(cache.statistics().insertions(), 0);
    ///
    /// cache.insert("a".to_string(), 1);
    /// assert_eq!(cache.statistics().insertions(), 1);
    ///
    /// cache.insert("a".to_string(), 2); // Update existing key
    /// assert_eq!(cache.statistics().insertions(), 1);
    /// # }
    /// ```
    pub fn insertions(&self) -> u64 {
        self.insertions
    }

    /// Returns the total number of cache hits.
    ///
    /// A cache hit occurs when a requested key is found in the cache.
    /// This metric is useful for measuring cache effectiveness.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(2).unwrap());
    /// cache.insert("a".to_string(), 1);
    ///
    /// assert_eq!(cache.statistics().hits(), 0);
    ///
    /// cache.get(&"a".to_string()); // Hit
    /// assert_eq!(cache.statistics().hits(), 1);
    ///
    /// cache.get(&"b".to_string()); // Miss
    /// assert_eq!(cache.statistics().hits(), 1);
    /// # }
    /// ```
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Returns the total number of cache misses.
    ///
    /// A cache miss occurs when a requested key is not found in the cache.
    /// High miss rates may indicate the cache size is too small or the access
    /// pattern is not well-suited for caching.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(2).unwrap());
    /// cache.insert("a".to_string(), 1);
    ///
    /// assert_eq!(cache.statistics().misses(), 0);
    ///
    /// cache.get(&"a".to_string()); // Hit
    /// assert_eq!(cache.statistics().misses(), 0);
    ///
    /// cache.get(&"b".to_string()); // Miss
    /// assert_eq!(cache.statistics().misses(), 1);
    /// # }
    /// ```
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Returns the cache hit rate as a percentage (0.0 to 100.0).
    ///
    /// The hit rate is calculated as `hits / (hits + misses) * 100`.
    /// A higher hit rate indicates better cache performance. Returns 0.0
    /// if no cache access operations have been performed yet.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(2).unwrap());
    /// cache.insert("a".to_string(), 1);
    /// cache.insert("b".to_string(), 2);
    ///
    /// // Initially no accesses, so hit rate is 0
    /// assert_eq!(cache.statistics().hit_rate(), 0.0);
    ///
    /// cache.get(&"a".to_string()); // Hit
    /// cache.get(&"b".to_string()); // Hit
    /// cache.get(&"c".to_string()); // Miss
    ///
    /// // 2 hits out of 3 total accesses = 66.67%
    /// assert!((cache.statistics().hit_rate() - 66.66666666666666).abs() < f64::EPSILON);
    /// # }
    /// ```
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64 * 100.0
        }
    }

    /// Returns the cache miss rate as a percentage (0.0 to 100.0).
    ///
    /// The miss rate is calculated as `misses / (hits + misses) * 100`.
    /// A lower miss rate indicates better cache performance. Returns 0.0
    /// if no cache access operations have been performed yet.
    ///
    /// Note: `hit_rate() + miss_rate()` always equals 100.0 (when there have
    /// been accesses).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "statistics")] {
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(std::num::NonZeroUsize::new(2).unwrap());
    /// cache.insert("a".to_string(), 1);
    ///
    /// // Initially no accesses, so miss rate is 0
    /// assert_eq!(cache.statistics().miss_rate(), 0.0);
    ///
    /// cache.get(&"a".to_string()); // Hit
    /// cache.get(&"b".to_string()); // Miss
    /// cache.get(&"c".to_string()); // Miss
    ///
    /// // 2 misses out of 3 total accesses = 66.67%
    /// assert!((cache.statistics().miss_rate() - 66.66666666666666).abs() < f64::EPSILON);
    /// # }
    /// ```
    pub fn miss_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.misses as f64 / (self.hits + self.misses) as f64 * 100.0
        }
    }
}

/// A generic cache implementation with configurable eviction policies.
///
/// `Cache` is the underlying generic structure that powers Lru, Mru, Lfu,
/// Fifo, Lifo, Random, and Sieve caches. The eviction behavior is determined by
/// the `PolicyType` parameter, which must implement the `Policy` trait.
///
/// For most use cases, you should use the type aliases [`Lru`], [`Mru`],
/// [`Lfu`], [`Fifo`], [`Lifo`], [`Random`], or [`Sieve`] instead of using
/// `Cache` directly.
///
/// # Type Parameters
///
/// * `Key` - The type of keys stored in the cache. Must implement [`Hash`] +
///   [`Eq`].
/// * `Value` - The type of values stored in the cache.
/// * `PolicyType` - The eviction policy implementation. Must implement
///   `Policy`.
///
/// # Memory Management
///
/// - Pre-allocates space for `capacity` entries to minimize reallocations
/// - Automatically evicts entries when capacity is exceeded
pub struct Cache<Key, Value, PolicyType: Policy<Value>> {
    queue: IndexMap<Key, <PolicyType::MetadataType as Metadata<Value>>::EntryType, RandomState>,
    capacity: NonZeroUsize,
    metadata: PolicyType::MetadataType,

    #[cfg(feature = "statistics")]
    statistics: Statistics,
}

impl<Key, Value, PolicyType: Policy<Value>> std::fmt::Debug for Cache<Key, Value, PolicyType>
where
    Key: std::fmt::Debug,
    Value: std::fmt::Debug,
    PolicyType::MetadataType: std::fmt::Debug,
    <PolicyType::MetadataType as Metadata<Value>>::EntryType: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache")
            .field("queue", &self.queue)
            .field("capacity", &self.capacity)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl<Key, Value, PolicyType: Policy<Value>> Clone for Cache<Key, Value, PolicyType>
where
    Key: Clone + Hash + Eq,
    Value: Clone,
    PolicyType::MetadataType: Clone,
    <PolicyType::MetadataType as Metadata<Value>>::EntryType: Clone,
{
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
            capacity: self.capacity,
            metadata: self.metadata.clone(),
            #[cfg(feature = "statistics")]
            statistics: self.statistics.clone(),
        }
    }
}

impl<Key: Hash + Eq, Value, PolicyType: Policy<Value>> Cache<Key, Value, PolicyType> {
    /// Creates a new, empty cache with the specified capacity.
    ///
    /// The cache will be able to hold at most `capacity` entries. When the
    /// cache is full and a new entry is inserted, the policy determines
    /// which existing entry will be evicted.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The maximum number of entries the cache can hold. Must be
    ///   greater than zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let cache: Lru<i32, String> = Lru::new(NonZeroUsize::new(100).unwrap());
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            queue: IndexMap::with_capacity_and_hasher(capacity.get(), RandomState::default()),
            capacity,
            metadata: PolicyType::MetadataType::default(),
            #[cfg(feature = "statistics")]
            statistics: Statistics::with_capacity(capacity),
        }
    }

    /// Returns the current statistics for the cache.
    #[cfg(feature = "statistics")]
    pub fn statistics(&self) -> Statistics {
        Statistics {
            len: self.queue.len(),
            capacity: self.capacity,
            ..self.statistics
        }
    }

    /// Resets the cache statistics to their initial state.
    #[cfg(feature = "statistics")]
    pub fn reset_statistics(&mut self) {
        self.statistics = Statistics::with_capacity(self.capacity);
    }

    /// Removes all entries from the cache.
    ///
    /// After calling this method, the cache will be empty and
    /// [`len()`](Self::len) will return 0. The capacity remains unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// assert_eq!(cache.len(), 2);
    /// cache.clear();
    /// assert_eq!(cache.len(), 0);
    /// assert!(cache.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.queue.clear();
        self.metadata = PolicyType::MetadataType::default();
    }

    /// Returns a reference to the value without updating its position in the
    /// cache.
    ///
    /// This method provides read-only access to a value without affecting the
    /// eviction order. Unlike [`get()`](Self::get), this will not mark the
    /// entry as touched.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up in the cache
    ///
    /// # Returns
    ///
    /// * `Some(&Value)` if the key exists in the cache
    /// * `None` if the key is not found
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// let previous_order = cache
    ///     .iter()
    ///     .map(|(k, v)| (*k, v.clone()))
    ///     .collect::<Vec<_>>();
    ///
    /// // Peek doesn't affect eviction order
    /// assert_eq!(cache.peek(&1), Some(&"one".to_string()));
    /// assert_eq!(cache.peek(&3), None);
    ///
    /// assert_eq!(cache.into_iter().collect::<Vec<_>>(), previous_order);
    /// ```
    pub fn peek(&self, key: &Key) -> Option<&Value> {
        self.queue.get(key).map(|entry| entry.value())
    }

    /// Returns a mutable reference to the value, updating the cache if the
    /// value is modified while borrowed.
    ///
    /// This method provides mutable access to a cached value through a smart
    /// `Entry` wrapper that automatically updates the cache's eviction
    /// order **only if** the value is actually modified during the borrow.
    /// Unlike [`get_mut()`](Self::get_mut), which always marks an entry as
    /// touched, `peek_mut` preserves the current eviction order unless changes
    /// are made to the value.
    ///
    /// The returned `Entry` acts as a smart pointer that:
    /// - Provides transparent access to the underlying value via
    ///   `Deref`/`DerefMut`
    /// - Tracks whether the value was modified during the borrow
    /// - Automatically updates the cache's eviction order on drop if
    ///   modifications occurred
    /// - Leaves the eviction order unchanged if no modifications were made
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up in the cache
    ///
    /// # Returns
    ///
    /// * `Some(`[`Entry<'_, Key, Value, PolicyType>`](crate::Entry)`)` - If the
    ///   key exists
    /// * `None` - If the key is not found in the cache
    ///
    /// # Behavior
    ///
    /// ## No Modification
    /// If you obtain an `Entry` but don't modify the value, the cache's
    /// eviction order remains unchanged:
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", vec![1, 2, 3]);
    /// cache.insert("B", vec![4, 5, 6]);
    ///
    /// let original_order: Vec<_> = cache.iter().map(|(k, _)| *k).collect();
    ///
    /// // Peek at the value without modifying it
    /// if let Some(entry) = cache.peek_mut(&"A") {
    ///     let _len = entry.len(); // Read-only access
    ///     // No modifications made
    /// }
    ///
    /// // Eviction order is preserved
    /// let new_order: Vec<_> = cache.iter().map(|(k, _)| *k).collect();
    /// assert_eq!(original_order, new_order);
    /// ```
    ///
    /// ## With Modification
    /// If you modify the value through the `Entry`, the cache automatically
    /// updates the eviction order when the `Entry` is dropped:
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", vec![1, 2, 3]);
    /// cache.insert("B", vec![4, 5, 6]);
    /// cache.insert("C", vec![7, 8, 9]);
    ///
    /// // Before: eviction order is A -> B -> C (A would be evicted first)
    /// assert_eq!(cache.tail().unwrap().0, &"A");
    ///
    /// // Modify "A" through peek_mut
    /// if let Some(mut entry) = cache.peek_mut(&"A") {
    ///     entry.push(4); // This modifies the value
    /// } // Entry is dropped here, triggering cache update
    ///
    /// // After: "A" is now most recently used, "B" would be evicted first
    /// assert_eq!(cache.tail().unwrap().0, &"B");
    /// ```
    ///
    /// # Performance
    ///
    /// - **Lookup**: O(1) average, O(n) worst case (same as hash map lookup)
    /// - **No modification**: No additional overhead beyond the lookup
    /// - **With modification**: O(1) cache reordering when the `Entry` is
    ///   dropped
    pub fn peek_mut(&'_ mut self, key: &Key) -> Option<Entry<'_, Key, Value, PolicyType>> {
        self.queue
            .get_index_of(key)
            .map(|index| Entry::new(index, self))
    }

    /// Returns a reference to the entry that would be evicted next.
    ///
    /// This method provides access to the entry at the "tail" of the eviction
    /// order without affecting the cache state. The specific entry returned
    /// depends on the policy:
    /// - Lru: Returns the least recently used entry
    /// - Mru: Returns the most recently used entry
    /// - Lfu: Returns the least frequently used entry
    /// - Fifo: Returns the first inserted entry
    /// - Lifo: Returns the last inserted entry
    /// - Random: Returns a randomly selected entry
    /// - Sieve: Returns the next entry that would be evicted after any second
    ///   chances. This may involve an O(n) scan across the cache to find the
    ///   next unvisited entry. **This does not update the visited bit, so an
    ///   eviction will need to rescan even if this method is called**.
    ///
    /// # Returns
    ///
    /// * `Some((&Key, &Value))` if the cache is not empty
    /// * `None` if the cache is empty
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// assert_eq!(cache.tail(), None);
    ///
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// // In Lru, tail returns the least recently used
    /// let (key, value) = cache.tail().unwrap();
    /// assert_eq!(key, &1);
    /// assert_eq!(value, &"one".to_string());
    /// ```
    pub fn tail(&self) -> Option<(&Key, &Value)> {
        self.queue
            .get_index(self.metadata.candidate_removal_index(&self.queue))
            .map(|(key, entry)| (key, entry.value()))
    }

    /// Returns true if the cache contains the given key.
    ///
    /// This method provides a quick way to check for key existence without
    /// affecting the eviction order or retrieving the value.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to check for existence
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    ///
    /// assert!(cache.contains_key(&1));
    /// assert!(!cache.contains_key(&2));
    /// ```
    pub fn contains_key(&self, key: &Key) -> bool {
        self.queue.contains_key(key)
    }

    /// Gets the value for a key, or inserts it using the provided function.
    ///
    /// If the key exists, returns a reference to the existing value and marks
    /// it as touched. If the key doesn't exist, calls the provided function to
    /// create a new value, inserts it, and returns a reference to it.
    ///
    /// When inserting into a full cache, the policy determines which entry is
    /// evicted.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up or insert
    /// * `or_insert` - Function called to create the value if the key doesn't
    ///   exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    ///
    /// // Insert new value
    /// let value = cache.get_or_insert_with(1, |&key| format!("value_{}", key));
    /// assert_eq!(value, "value_1");
    ///
    /// // Get existing value (function not called)
    /// let value = cache.get_or_insert_with(1, |&key| format!("different_{}", key));
    /// assert_eq!(value, "value_1");
    /// ```
    pub fn get_or_insert_with(
        &mut self,
        key: Key,
        or_insert: impl FnOnce(&Key) -> Value,
    ) -> &Value {
        self.get_or_insert_with_mut(key, or_insert)
    }

    /// Gets the value for a key, or inserts it using the provided function.
    ///
    /// This is the mutable version of
    /// [`get_or_insert_with()`](Self::get_or_insert_with). If the key
    /// exists, returns a mutable reference to the existing value and marks
    /// it as touched. If the key doesn't exist, calls the provided function to
    /// create a new value, inserts it, and returns a mutable reference to
    /// it.
    ///
    /// When inserting into a full cache, the policy determines which entry is
    /// evicted.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up or insert
    /// * `or_insert` - Function called to create the value if the key doesn't
    ///   exist
    ///
    /// # Returns
    ///
    /// A mutable reference to the value (existing or newly inserted).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    ///
    /// // Insert new value and modify it
    /// let value = cache.get_or_insert_with_mut(1, |&key| format!("value_{}", key));
    /// value.push_str("_modified");
    /// assert_eq!(cache.peek(&1), Some(&"value_1_modified".to_string()));
    ///
    /// // Get existing value and modify it further (function not called)
    /// let value = cache.get_or_insert_with_mut(1, |&key| format!("different_{}", key));
    /// value.push_str("_again");
    /// assert_eq!(cache.peek(&1), Some(&"value_1_modified_again".to_string()));
    /// ```
    pub fn get_or_insert_with_mut(
        &mut self,
        key: Key,
        or_insert: impl FnOnce(&Key) -> Value,
    ) -> &mut Value {
        let len = self.queue.len();
        match self.queue.entry(key) {
            indexmap::map::Entry::Occupied(o) => {
                let index =
                    PolicyType::touch_entry(o.index(), false, &mut self.metadata, &mut self.queue);

                #[cfg(feature = "statistics")]
                {
                    self.statistics.hits += 1;
                }

                self.queue[index].value_mut()
            }
            indexmap::map::Entry::Vacant(v) => {
                let index = v.index();
                let e = <PolicyType::MetadataType as Metadata<Value>>::EntryType::new(or_insert(
                    v.key(),
                ));
                v.insert(e);
                #[cfg(feature = "statistics")]
                {
                    self.statistics.insertions += 1;
                    self.statistics.misses += 1;

                    self.statistics.evictions += u64::from(len == self.capacity.get());
                }

                let index = PolicyType::touch_entry(
                    index,
                    len == self.capacity.get(),
                    &mut self.metadata,
                    &mut self.queue,
                );
                self.queue[index].value_mut()
            }
        }
    }

    /// Gets a value from the cache, marking it as touched. If the key is not
    /// present, inserts the default value and returns a reference to it.
    ///
    /// This method will trigger eviction if the cache is at capacity and the
    /// key is not already present.
    ///
    /// # Type Requirements
    ///
    /// The value type must implement [`Default`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, Vec<String>>::new(NonZeroUsize::new(2).unwrap());
    ///
    /// // Get or insert default (empty Vec<String>)
    /// let vec_ref = cache.get_or_default(1);
    /// assert!(vec_ref.is_empty());
    ///
    /// // Modify the value through another method
    /// cache.get_mut(&1).unwrap().push("hello".to_string());
    ///
    /// // Subsequent calls return the existing value
    /// let vec_ref = cache.get_or_default(1);
    /// assert_eq!(vec_ref.len(), 1);
    /// assert_eq!(vec_ref[0], "hello");
    /// ```
    pub fn get_or_default(&mut self, key: Key) -> &Value
    where
        Value: Default,
    {
        self.get_or_insert_with(key, |_| Value::default())
    }

    /// Gets a mutable value from the cache, marking it as touched. If the key
    /// is not present, inserts the default value and returns a mutable
    /// reference to it.
    ///
    /// This method will trigger eviction if the cache is at capacity and the
    /// key is not already present.
    ///
    /// # Type Requirements
    ///
    /// The value type must implement [`Default`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, Vec<String>>::new(NonZeroUsize::new(2).unwrap());
    ///
    /// // Get or insert default and modify it immediately
    /// let vec_ref = cache.get_or_default_mut(1);
    /// vec_ref.push("hello".to_string());
    /// vec_ref.push("world".to_string());
    ///
    /// // Verify the changes were made
    /// assert_eq!(cache.get(&1).unwrap().len(), 2);
    /// assert_eq!(cache.get(&1).unwrap()[0], "hello");
    ///
    /// // Subsequent calls return the existing mutable value
    /// let vec_ref = cache.get_or_default_mut(1);
    /// vec_ref.clear();
    /// assert!(cache.get(&1).unwrap().is_empty());
    /// ```
    pub fn get_or_default_mut(&mut self, key: Key) -> &mut Value
    where
        Value: Default,
    {
        self.get_or_insert_with_mut(key, |_| Value::default())
    }

    /// The immutable version of `insert_mut`. See [`Self::insert_mut`] for
    /// details.
    pub fn insert(&mut self, key: Key, value: Value) -> &Value {
        self.insert_mut(key, value)
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// This is the mutable version of [`insert()`](Self::insert). If the key
    /// already exists, its value is updated and the entry is marked as touched.
    /// If the key is new and the cache is at capacity, the policy determines
    /// which entry is evicted to make room.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to associate with the key
    ///
    /// # Returns
    ///
    /// A mutable reference to the inserted value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(2).unwrap());
    ///
    /// // Insert new value and modify it immediately
    /// let value = cache.insert_mut(1, "one".to_string());
    /// value.push_str("_modified");
    /// assert_eq!(cache.peek(&1), Some(&"one_modified".to_string()));
    ///
    /// // Update existing value
    /// let value = cache.insert_mut(1, "new_one".to_string());
    /// value.push_str("_updated");
    /// assert_eq!(cache.peek(&1), Some(&"new_one_updated".to_string()));
    ///
    /// // Insert when at capacity (evicts based on policy)
    /// cache.insert_mut(2, "two".to_string());
    /// let value = cache.insert_mut(3, "three".to_string());
    /// value.push_str("_latest");
    /// assert_eq!(
    ///     cache.into_iter().collect::<Vec<_>>(),
    ///     [(2, "two".to_string()), (3, "three_latest".to_string())]
    /// );
    /// ```
    pub fn insert_mut(&mut self, key: Key, value: Value) -> &mut Value {
        let len = self.queue.len();
        match self.queue.entry(key) {
            indexmap::map::Entry::Occupied(mut o) => {
                *o.get_mut().value_mut() = value;
                let index =
                    PolicyType::touch_entry(o.index(), false, &mut self.metadata, &mut self.queue);

                #[cfg(feature = "statistics")]
                {
                    self.statistics.hits += 1;
                }

                self.queue[index].value_mut()
            }
            indexmap::map::Entry::Vacant(v) => {
                let index = v.index();
                v.insert(<PolicyType::MetadataType as Metadata<Value>>::EntryType::new(value));

                let index = PolicyType::touch_entry(
                    index,
                    len == self.capacity.get(),
                    &mut self.metadata,
                    &mut self.queue,
                );

                #[cfg(feature = "statistics")]
                {
                    self.statistics.insertions += 1;
                    self.statistics.evictions += u64::from(len == self.capacity.get());
                }

                self.queue[index].value_mut()
            }
        }
    }

    /// The immutable version of `try_insert_or_update_mut`. See
    /// [`Self::try_insert_or_update_mut`] for details.
    pub fn try_insert_or_update(&mut self, key: Key, value: Value) -> Result<&Value, (Key, Value)> {
        self.try_insert_or_update_mut(key, value).map(|v| &*v)
    }

    /// Attempts to insert a key-value pair into the cache without triggering
    /// eviction, returning a mutable reference to the inserted value.
    ///
    /// This method only inserts the entry if the cache has available capacity.
    /// If the cache is at capacity, the insertion fails and the key-value pair
    /// is returned unchanged. Unlike [`insert_mut`](Self::insert_mut), this
    /// method will never evict existing entries.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert
    /// * `value` - The value to associate with the key
    ///
    /// # Returns
    ///
    /// * `Ok(&mut Value)` - A mutable reference to the inserted value if
    ///   successful
    /// * `Err((Key, Value))` - The original key-value pair if insertion failed
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, Vec<String>>::new(NonZeroUsize::new(2).unwrap());
    ///
    /// // Successful insertion with immediate mutation
    /// let vec_ref = cache
    ///     .try_insert_or_update_mut(1, vec!["hello".to_string()])
    ///     .unwrap();
    /// vec_ref.push("world".to_string());
    /// assert_eq!(cache.get(&1).unwrap().len(), 2);
    ///
    /// // Fill remaining capacity
    /// cache
    ///     .try_insert_or_update_mut(2, vec!["foo".to_string()])
    ///     .unwrap();
    ///
    /// // Failed insertion when cache is at capacity
    /// let result = cache.try_insert_or_update_mut(3, vec!["bar".to_string()]);
    /// assert!(result.is_err());
    /// if let Err((key, value)) = result {
    ///     assert_eq!(key, 3);
    ///     assert_eq!(value, vec!["bar".to_string()]);
    /// }
    /// assert_eq!(cache.len(), 2); // Cache unchanged
    /// ```
    ///
    /// # Updating Existing Keys
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, Vec<i32>>::new(NonZeroUsize::new(1).unwrap());
    /// cache.insert(1, vec![1, 2, 3]);
    ///
    /// // Updating an existing key always succeeds and allows mutation
    /// let vec_ref = cache.try_insert_or_update_mut(1, vec![4, 5]).unwrap();
    /// vec_ref.push(6);
    /// assert_eq!(cache.get(&1).unwrap(), &vec![4, 5, 6]);
    /// ```
    pub fn try_insert_or_update_mut(
        &mut self,
        key: Key,
        value: Value,
    ) -> Result<&mut Value, (Key, Value)> {
        let len = self.queue.len();
        match self.queue.entry(key) {
            indexmap::map::Entry::Occupied(mut o) => {
                *o.get_mut().value_mut() = value;
                let index =
                    PolicyType::touch_entry(o.index(), false, &mut self.metadata, &mut self.queue);

                #[cfg(feature = "statistics")]
                {
                    self.statistics.hits += 1;
                }

                Ok(self.queue[index].value_mut())
            }
            indexmap::map::Entry::Vacant(v) => {
                #[cfg(feature = "statistics")]
                {
                    self.statistics.misses += 1;
                }

                if len >= self.capacity.get() {
                    return Err((v.into_key(), value));
                }

                #[cfg(feature = "statistics")]
                {
                    self.statistics.insertions += 1;
                }

                let index = v.index();
                v.insert(<PolicyType::MetadataType as Metadata<Value>>::EntryType::new(value));
                let index =
                    PolicyType::touch_entry(index, false, &mut self.metadata, &mut self.queue);
                Ok(self.queue[index].value_mut())
            }
        }
    }

    /// The immutable version of `try_get_or_insert_with_mut`. See
    /// [`Self::try_get_or_insert_with_mut`] for details.
    pub fn try_get_or_insert_with(
        &mut self,
        key: Key,
        or_insert: impl FnOnce(&Key) -> Value,
    ) -> Result<&Value, Key> {
        self.try_get_or_insert_with_mut(key, or_insert).map(|v| &*v)
    }

    /// Attempts to get a mutable value from the cache or insert it using a
    /// closure, without triggering eviction of other entries.
    ///
    /// If the key exists, returns a mutable reference to the existing value and
    /// marks it as touched. If the key doesn't exist and the cache has
    /// available capacity, the closure is called to generate a value which
    /// is then inserted. If the cache is at capacity and the key doesn't
    /// exist, the operation fails and returns the key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up or insert
    /// * `or_insert` - A closure that generates the value if the key is not
    ///   found
    ///
    /// # Returns
    ///
    /// * `Ok(&mut Value)` - A mutable reference to the value if successful
    /// * `Err(Key)` - The original key if insertion failed due to capacity
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, Vec<String>>::new(NonZeroUsize::new(2).unwrap());
    ///
    /// // Insert new value and modify it immediately
    /// let vec_ref = cache
    ///     .try_get_or_insert_with_mut(1, |_| vec!["initial".to_string()])
    ///     .unwrap();
    /// vec_ref.push("added".to_string());
    /// assert_eq!(cache.get(&1).unwrap().len(), 2);
    ///
    /// // Get existing value and modify it further
    /// let vec_ref = cache
    ///     .try_get_or_insert_with_mut(1, |_| vec!["not-called".to_string()])
    ///     .unwrap();
    /// vec_ref.push("more".to_string());
    /// assert_eq!(cache.get(&1).unwrap().len(), 3);
    ///
    /// // Fill remaining capacity
    /// cache
    ///     .try_get_or_insert_with_mut(2, |_| vec!["second".to_string()])
    ///     .unwrap();
    ///
    /// // Fail when cache is at capacity and key doesn't exist
    /// let result = cache.try_get_or_insert_with_mut(3, |_| vec!["third".to_string()]);
    /// assert!(result.is_err());
    /// assert_eq!(result.unwrap_err(), 3);
    /// assert_eq!(cache.len(), 2); // Cache unchanged
    /// ```
    ///
    /// # Use Cases
    ///
    /// This method is particularly useful for scenarios where you need to:
    /// - Initialize complex data structures in-place
    /// - Accumulate data in existing cache entries
    /// - Avoid unnecessary allocations when the entry already exists
    ///
    /// ```rust
    /// use std::{
    ///     collections::HashMap,
    ///     num::NonZeroUsize,
    /// };
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<String, HashMap<String, i32>>::new(NonZeroUsize::new(3).unwrap());
    ///
    /// // Build up a map incrementally
    /// let map_ref = cache
    ///     .try_get_or_insert_with_mut("counters".to_string(), |_| HashMap::new())
    ///     .unwrap();
    /// map_ref.insert("clicks".to_string(), 1);
    /// map_ref.insert("views".to_string(), 5);
    ///
    /// // Later, update the existing map
    /// let map_ref = cache
    ///     .try_get_or_insert_with_mut(
    ///         "counters".to_string(),
    ///         |_| HashMap::new(), // Not called since key exists
    ///     )
    ///     .unwrap();
    /// *map_ref.get_mut("clicks").unwrap() += 1;
    /// ```
    pub fn try_get_or_insert_with_mut(
        &mut self,
        key: Key,
        or_insert: impl FnOnce(&Key) -> Value,
    ) -> Result<&mut Value, Key> {
        let len = self.queue.len();
        match self.queue.entry(key) {
            indexmap::map::Entry::Occupied(o) => {
                let index =
                    PolicyType::touch_entry(o.index(), false, &mut self.metadata, &mut self.queue);

                #[cfg(feature = "statistics")]
                {
                    self.statistics.hits += 1;
                }

                Ok(self.queue[index].value_mut())
            }
            indexmap::map::Entry::Vacant(v) => {
                #[cfg(feature = "statistics")]
                {
                    self.statistics.misses += 1;
                }

                if len >= self.capacity.get() {
                    return Err(v.into_key());
                }

                #[cfg(feature = "statistics")]
                {
                    self.statistics.insertions += 1;
                }

                let index = v.index();
                let e = <PolicyType::MetadataType as Metadata<Value>>::EntryType::new(or_insert(
                    v.key(),
                ));
                v.insert(e);
                let index =
                    PolicyType::touch_entry(index, false, &mut self.metadata, &mut self.queue);
                Ok(self.queue[index].value_mut())
            }
        }
    }

    /// The immutable version of `get_mut`. See [`Self::get_mut`] for details.
    pub fn get(&mut self, key: &Key) -> Option<&Value> {
        self.get_mut(key).map(|v| &*v)
    }

    /// Gets a value from the cache, marking it as touched.
    ///
    /// This is the mutable version of [`get()`](Self::get). If the key exists,
    /// returns a mutable reference to the value and updates the entry's
    /// position in the eviction order according to the policy. If the key
    /// doesn't exist, returns `None` and the cache is unchanged.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// * `Some(&mut Value)` if the key exists
    /// * `None` if the key doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// // Get and modify the value, marking as recently used
    /// if let Some(value) = cache.get_mut(&1) {
    ///     value.push_str("_modified");
    /// }
    /// assert_eq!(cache.peek(&1), Some(&"one_modified".to_string()));
    ///
    /// // Non-existent key returns None
    /// assert_eq!(cache.get_mut(&3), None);
    ///
    /// assert_eq!(
    ///     cache.into_iter().collect::<Vec<_>>(),
    ///     vec![(2, "two".to_string()), (1, "one_modified".to_string())]
    /// );
    /// ```
    pub fn get_mut(&mut self, key: &Key) -> Option<&mut Value> {
        if let Some(index) = self.queue.get_index_of(key) {
            let index = PolicyType::touch_entry(index, false, &mut self.metadata, &mut self.queue);

            #[cfg(feature = "statistics")]
            {
                self.statistics.hits += 1;
            }

            Some(self.queue[index].value_mut())
        } else {
            #[cfg(feature = "statistics")]
            {
                self.statistics.misses += 1;
            }
            None
        }
    }

    /// Removes and returns the entry that would be evicted next.
    ///
    /// This method removes and returns the entry at the "tail" of the eviction
    /// order. The specific entry removed depends on the policy:
    /// - Lru: Removes the least recently used entry
    /// - Mru: Removes the most recently used entry
    /// - Lfu: Removes the least frequently used entry
    /// - Fifo: Removes the first inserted entry
    /// - Lifo: Removes the last inserted entry
    /// - Random: Removes a randomly selected entry
    ///
    /// # Returns
    ///
    /// * `Some((Key, Value))` if the cache is not empty
    /// * `None` if the cache is empty
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// // Pop the least recently used entry
    /// let (key, value) = cache.pop().unwrap();
    /// assert_eq!(key, 1);
    /// assert_eq!(value, "one".to_string());
    /// assert_eq!(
    ///     cache.into_iter().collect::<Vec<_>>(),
    ///     vec![(2, "two".to_string())]
    /// );
    /// ```
    pub fn pop(&mut self) -> Option<(Key, Value)> {
        PolicyType::evict_entry(&mut self.metadata, &mut self.queue)
            .1
            .map(|(key, entry)| {
                #[cfg(feature = "statistics")]
                {
                    self.statistics.evictions += 1;
                }

                (key, entry.into_value())
            })
    }

    /// Removes a specific entry from the cache.
    ///
    /// If the key exists, removes it from the cache and returns the associated
    /// value. If the key doesn't exist, returns `None` and the cache is
    /// unchanged.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to remove from the cache
    ///
    /// # Returns
    ///
    /// * `Some(Value)` if the key existed and was removed
    /// * `None` if the key was not found
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// assert_eq!(cache.remove(&1), Some("one".to_string()));
    /// assert_eq!(cache.remove(&1), None); // Already removed
    /// assert_eq!(
    ///     cache.into_iter().collect::<Vec<_>>(),
    ///     vec![(2, "two".to_string())]
    /// );
    /// ```
    pub fn remove(&mut self, key: &Key) -> Option<Value> {
        if let Some(index) = self.queue.get_index_of(key) {
            PolicyType::swap_remove_entry(index, &mut self.metadata, &mut self.queue)
                .map(|(_, entry)| entry.into_value())
        } else {
            None
        }
    }

    /// Returns true if the cache contains no entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// assert!(cache.is_empty());
    ///
    /// cache.insert(1, "one".to_string());
    /// assert!(!cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the number of entries currently in the cache.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// assert_eq!(cache.len(), 0);
    ///
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    /// assert_eq!(cache.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns the maximum number of entries the cache can hold.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let cache = Lru::<i32, String>::new(NonZeroUsize::new(100).unwrap());
    /// assert_eq!(cache.capacity(), 100);
    /// assert_eq!(cache.len(), 0); // Empty but has capacity for 100
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity.get()
    }

    /// Sets a new capacity for the cache.
    ///
    /// If the new capacity is smaller than the current number of entries,
    /// entries will be evicted according to the cache's eviction policy
    /// until the cache size fits within the new capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The new maximum number of entries the cache can hold.
    ///   Must be greater than zero.
    ///
    /// # Returns
    ///
    /// A vector of evicted entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    /// cache.insert(3, "three".to_string());
    /// assert_eq!(cache.len(), 3);
    /// assert_eq!(cache.capacity(), 3);
    ///
    /// // Increase capacity
    /// cache.set_capacity(NonZeroUsize::new(5).unwrap());
    /// assert_eq!(cache.capacity(), 5);
    /// assert_eq!(cache.len(), 3); // Existing entries remain
    ///
    /// // Decrease capacity - will evict entries
    /// cache.set_capacity(NonZeroUsize::new(2).unwrap());
    /// assert_eq!(cache.capacity(), 2);
    /// assert_eq!(cache.len(), 2); // One entry was evicted
    /// ```
    pub fn set_capacity(&mut self, capacity: NonZeroUsize) -> Vec<(Key, Value)> {
        let mut evicted = Vec::with_capacity(self.queue.len().saturating_sub(capacity.get()));
        if capacity.get() < self.queue.len() {
            // If new capacity is smaller than current size, we need to evict
            // entries until we fit
            while self.queue.len() > capacity.get()
                && let Some(kv) = self.pop()
            {
                evicted.push(kv);
            }
        }

        self.capacity = capacity;
        evicted
    }

    /// Evicts entries from the cache until the size is reduced to the specified
    /// number.
    ///
    /// This method removes entries according to the cache's eviction policy
    /// until the cache contains at most `n` entries. If the cache already
    /// has `n` or fewer entries, no eviction occurs.
    ///
    /// # Arguments
    ///
    /// * `n` - The target number of entries to keep in the cache
    ///
    /// # Returns
    ///
    /// A vector containing all evicted key-value pairs in the order they were
    /// removed from the cache.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(5).unwrap());
    ///
    /// // Fill the cache
    /// for i in 1..=5 {
    ///     cache.insert(i, format!("value{}", i));
    /// }
    /// assert_eq!(cache.len(), 5);
    ///
    /// // Evict until only 2 entries remain
    /// let evicted = cache.evict_to_size(2);
    /// assert_eq!(cache.len(), 2);
    /// assert_eq!(evicted.len(), 3);
    ///
    /// // The evicted entries follow the cache's eviction policy
    /// // For LRU, oldest entries are evicted first
    /// assert_eq!(evicted[0], (1, "value1".to_string()));
    /// assert_eq!(evicted[1], (2, "value2".to_string()));
    /// assert_eq!(evicted[2], (3, "value3".to_string()));
    ///
    /// // Cache now contains the most recently used entries
    /// assert!(cache.contains_key(&4));
    /// assert!(cache.contains_key(&5));
    /// ```
    ///
    /// # Behavior with Different Cache Sizes
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// // No eviction when target size >= current size
    /// let evicted = cache.evict_to_size(5);
    /// assert_eq!(evicted.len(), 0);
    /// assert_eq!(cache.len(), 2);
    ///
    /// // Evict to exact current size
    /// let evicted = cache.evict_to_size(2);
    /// assert_eq!(evicted.len(), 0);
    /// assert_eq!(cache.len(), 2);
    ///
    /// // Evict to zero (clear cache)
    /// let evicted = cache.evict_to_size(0);
    /// assert_eq!(evicted.len(), 2);
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn evict_to_size(&mut self, n: usize) -> Vec<(Key, Value)> {
        let mut evicted = Vec::with_capacity(n);
        while self.queue.len() > n
            && let Some(kv) = self.pop()
        {
            evicted.push(kv);
        }
        evicted
    }

    /// Evicts up to `n` entries from the cache according to the eviction
    /// policy.
    ///
    /// This method removes at most `n` entries from the cache, following the
    /// cache's eviction policy. If the cache contains fewer than `n` entries,
    /// all remaining entries are evicted.
    ///
    /// # Arguments
    ///
    /// * `n` - The maximum number of entries to evict
    ///
    /// # Returns
    ///
    /// A vector containing all evicted key-value pairs in the order they were
    /// removed from the cache. The vector will contain at most `n` entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(5).unwrap());
    ///
    /// // Fill the cache
    /// for i in 1..=5 {
    ///     cache.insert(i, format!("value{}", i));
    /// }
    /// assert_eq!(cache.len(), 5);
    ///
    /// // Evict exactly 3 entries
    /// let evicted = cache.evict_n_entries(3);
    /// assert_eq!(cache.len(), 2);
    /// assert_eq!(evicted.len(), 3);
    ///
    /// // For LRU, oldest entries are evicted first
    /// assert_eq!(evicted[0], (1, "value1".to_string()));
    /// assert_eq!(evicted[1], (2, "value2".to_string()));
    /// assert_eq!(evicted[2], (3, "value3".to_string()));
    ///
    /// // Cache retains the most recently used entries
    /// assert!(cache.contains_key(&4));
    /// assert!(cache.contains_key(&5));
    /// ```
    ///
    /// # Behavior with Edge Cases
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// // Requesting more entries than available
    /// let evicted = cache.evict_n_entries(5);
    /// assert_eq!(evicted.len(), 2); // Only 2 entries were available
    /// assert_eq!(cache.len(), 0); // Cache is now empty
    ///
    /// // Evicting from empty cache
    /// let evicted = cache.evict_n_entries(3);
    /// assert_eq!(evicted.len(), 0); // No entries to evict
    /// assert_eq!(cache.len(), 0);
    ///
    /// // Evicting zero entries
    /// cache.insert(1, "one".to_string());
    /// let evicted = cache.evict_n_entries(0);
    /// assert_eq!(evicted.len(), 0); // No eviction requested
    /// assert_eq!(cache.len(), 1); // Cache unchanged
    /// ```
    pub fn evict_n_entries(&mut self, n: usize) -> Vec<(Key, Value)> {
        let mut evicted = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(kv) = self.pop() {
                evicted.push(kv);
            } else {
                break; // No more entries to evict
            }
        }
        evicted
    }

    /// Returns an iterator over cache entries in eviction order.
    ///
    /// The iterator yields key-value pairs `(&Key, &Value)` in the order they
    /// would be evicted from the cache, starting with the entry that would be
    /// evicted first. This allows you to inspect the cache's internal ordering
    /// and understand which entries are most at risk of eviction. This does not
    /// permute the order of the cache.
    ///
    /// The specific iteration order depends on the cache's eviction policy:
    ///
    /// ## Lru (Least Recently Used)
    /// Iterates from **least recently used** to **most recently used**:
    /// - First yielded: The entry accessed longest ago (would be evicted first)
    /// - Last yielded: The entry accessed most recently (would be evicted last)
    ///
    /// ## Mru (Most Recently Used)  
    /// Iterates from **most recently used** to **least recently used**:
    /// - First yielded: The entry accessed most recently (would be evicted
    ///   first)
    /// - Last yielded: The entry accessed longest ago (would be evicted last)
    ///
    /// ## Lfu (Least Frequently Used)
    /// Iterates from **least frequently used** to **most frequently used**:
    /// - Entries are ordered by access frequency (lower frequency first)
    /// - Ties are broken by insertion/access order within frequency buckets
    /// - First yielded: Entry with lowest access count (would be evicted first)
    /// - Last yielded: Entry with highest access count (would be evicted last)
    ///
    /// ## Fifo (First In, First Out)
    /// Iterates from **first inserted** to **last inserted**:
    /// - Entries are ordered by insertion time
    /// - First yielded: Entry inserted earliest (would be evicted first)
    /// - Last yielded: Entry inserted most recently (would be evicted last)
    /// - Access patterns do not affect iteration order
    ///
    /// ## Lifo (Last In, First Out)
    /// Iterates from **last inserted** to **first inserted**:
    /// - Entries are ordered by insertion time in reverse
    /// - First yielded: Entry inserted most recently (would be evicted first)
    /// - Last yielded: Entry inserted earliest (would be evicted last)
    /// - Access patterns do not affect iteration order
    ///
    /// ## Random
    /// Iterates in **random order** in debug builds, **arbitrary order** in
    /// release builds:
    /// - Debug builds: Order is randomized for testing purposes
    /// - Release builds: Order is arbitrary and depends on the pattern of
    ///   eviction and insertion.
    /// - The eviction order itself is always random regardless of iteration
    ///   order
    ///
    /// ## Sieve
    /// Iterates in **eviction order**:
    /// - Starts from the current hand position (next eviction candidate)
    /// - Simulates the eviction scanning process: skips visited entries once,
    ///   then includes them in order
    /// - Uses additional state tracking to maintain this logical ordering
    /// - **Higher iteration overhead** than other policies due to eviction
    ///   simulation
    ///
    /// # Time Complexity
    /// - Creating the iterator: **O(1)**
    /// - Iterating through all entries: **O(n)** where n is the cache size
    /// - Each `next()` call: **O(1)** for Lru/Mru/Fifo/Lifo/Random, **O(1)
    ///   amortized** for Lfu, **O(1) with higher constant overhead** for Sieve
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", 1);
    /// cache.insert("B", 2);
    /// cache.insert("C", 3);
    ///
    /// // Lru: Iterates from least recently used to most recently used
    /// let items: Vec<_> = cache.iter().collect();
    /// assert_eq!(items, [(&"A", &1), (&"B", &2), (&"C", &3)]);
    /// ```
    ///
    /// ## After Access Pattern
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", 1);
    /// cache.insert("B", 2);
    /// cache.insert("C", 3);
    ///
    /// // Access "A" to make it recently used
    /// cache.get(&"A");
    ///
    /// // Now "B" is least recently used, "A" is most recently used
    /// let items: Vec<_> = cache.iter().collect();
    /// assert_eq!(items, [(&"B", &2), (&"C", &3), (&"A", &1)]);
    /// ```
    ///
    /// ## Mru Policy Example
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Mru;
    ///
    /// let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", 1);
    /// cache.insert("B", 2);
    /// cache.insert("C", 3);
    ///
    /// // Mru: Iterates from most recently used to least recently used
    /// let items: Vec<_> = cache.iter().collect();
    /// assert_eq!(items, [(&"C", &3), (&"B", &2), (&"A", &1)]);
    /// ```
    ///
    /// ## Lfu Policy Example
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lfu;
    ///
    /// let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", 1);
    /// cache.insert("B", 2);
    /// cache.insert("C", 3);
    ///
    /// // Access "A" multiple times to increase its frequency
    /// cache.get(&"A");
    /// cache.get(&"A");
    /// cache.get(&"B"); // Access "B" once
    ///
    /// // Lfu: "C" (freq=0), "B" (freq=1), "A" (freq=2)
    /// let items: Vec<_> = cache.iter().collect();
    /// assert_eq!(items, [(&"C", &3), (&"B", &2), (&"A", &1)]);
    /// ```
    ///
    /// ## Consistency with `tail()`
    /// The first item from the iterator always matches `tail()` with the
    /// exception of the `Random` policy:
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", 1);
    /// cache.insert("B", 2);
    ///
    /// let tail_item = cache.tail();
    /// let first_iter_item = cache.iter().next();
    /// assert_eq!(tail_item, first_iter_item);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&Key, &Value)> {
        PolicyType::iter(&self.metadata, &self.queue)
    }

    /// Returns an iterator over the keys in the cache in eviction order.
    ///
    /// This method returns an iterator that yields references to all keys
    /// currently stored in the cache. The keys are returned in the same order
    /// as [`iter()`](Self::iter), which follows the cache's eviction policy.
    ///
    /// # Eviction Order
    ///
    /// The iteration order depends on the cache's eviction policy:
    /// - **Lru**: Keys from least recently used to most recently used
    /// - **Mru**: Keys from most recently used to least recently used
    /// - **Lfu**: Keys from least frequently used to most frequently used
    /// - **Fifo**: Keys from first inserted to last inserted
    /// - **Lifo**: Keys from last inserted to first inserted
    /// - **Random**: Keys in random order (debug) or arbitrary order (release)
    /// - **Sieve**: Keys in eviction order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", 1);
    /// cache.insert("B", 2);
    /// cache.insert("C", 3);
    ///
    /// // Get keys in Lru order (least recently used first)
    /// let keys: Vec<_> = cache.keys().collect();
    /// assert_eq!(keys, [&"A", &"B", &"C"]);
    /// ```
    pub fn keys(&self) -> impl Iterator<Item = &Key> {
        PolicyType::iter(&self.metadata, &self.queue).map(|(k, _)| k)
    }

    /// Returns an iterator over the values in the cache in eviction order.
    ///
    /// This method returns an iterator that yields references to all values
    /// currently stored in the cache. The values are returned in the same order
    /// as [`iter()`](Self::iter), which follows the cache's eviction policy.
    ///
    /// # Eviction Order
    ///
    /// The iteration order depends on the cache's eviction policy:
    /// - **Lru**: Values from least recently used to most recently used
    /// - **Mru**: Values from most recently used to least recently used
    /// - **Lfu**: Values from least frequently used to most frequently used
    /// - **Fifo**: Values from first inserted to last inserted
    /// - **Lifo**: Values from last inserted to first inserted
    /// - **Random**: Values in random order (debug) or arbitrary order
    ///   (release)
    /// - **Sieve**: Values following eviction order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("A", 1);
    /// cache.insert("B", 2);
    /// cache.insert("C", 3);
    ///
    /// // Get values in Lru order (least recently used first)
    /// let values: Vec<_> = cache.values().collect();
    /// assert_eq!(values, [&1, &2, &3]);
    /// ```
    pub fn values(&self) -> impl Iterator<Item = &Value> {
        PolicyType::iter(&self.metadata, &self.queue).map(|(_, v)| v)
    }

    /// Retains only the entries for which the predicate returns `true`.
    ///
    /// This iterates in **arbitrary order**. I.e. unlike `iter()`, the order of
    /// items you see is not guaranteed to match the eviction order.
    ///
    /// This method removes all entries from the cache for which the provided
    /// closure returns `false`. The predicate is called with a reference to
    /// each key and a mutable reference to each value, allowing for both
    /// filtering based on key/value content and in-place modification of
    /// retained values.
    ///
    /// The operation preserves the relative order of retained entries according
    /// to their original insertion order. This operation does not perturb the
    /// eviction order, with the exception of entries that are removed.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that takes `(&Key, &mut Value)` and returns `true` for
    ///   entries that should be kept, `false` for entries that should be
    ///   removed
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(5).unwrap());
    /// cache.insert(1, "apple".to_string());
    /// cache.insert(2, "banana".to_string());
    /// cache.insert(3, "cherry".to_string());
    /// cache.insert(4, "date".to_string());
    ///
    /// // Keep only entries where the key is even
    /// cache.retain(|&key, _value| key % 2 == 0);
    /// assert_eq!(
    ///     cache.into_iter().collect::<Vec<_>>(),
    ///     [(2, "banana".to_string()), (4, "date".to_string())]
    /// );
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&Key, &mut Value) -> bool,
    {
        let mut offset = 0;
        for idx in 0..self.queue.len() {
            let Some((k, v)) = self.queue.get_index_mut(idx - offset) else {
                break;
            };
            if !f(k, v.value_mut()) {
                PolicyType::swap_remove_entry(idx - offset, &mut self.metadata, &mut self.queue);
                offset += 1;
            }
        }
    }

    /// Shrinks the internal storage to fit the current number of entries.
    pub fn shrink_to_fit(&mut self) {
        self.queue.shrink_to_fit();
    }
}

impl<K: Hash + Eq + std::fmt::Debug, Value: std::fmt::Debug, PolicyType: Policy<Value>>
    Cache<K, Value, PolicyType>
{
    #[doc(hidden)]
    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    pub fn debug_validate(&self) {
        PolicyType::debug_validate(&self.metadata, &self.queue);
    }
}

impl<K, V, PolicyType: Policy<V>> IntoIterator for Cache<K, V, PolicyType> {
    type IntoIter = PolicyType::IntoIter<K>;
    type Item = (K, V);

    /// Converts the cache into an iterator over key-value pairs.
    ///
    /// This method implements the standard library's `IntoIterator` trait,
    /// allowing caches to be consumed and converted into iterators. The
    /// iteration order follows the eviction policy's ordering:
    ///
    /// - **Lru**: From least recently used to most recently used
    /// - **Mru**: From most recently used to least recently used
    /// - **Lfu**: From least frequently used to most frequently used
    /// - **Fifo**: From first inserted to last inserted
    /// - **Lifo**: From last inserted to first inserted
    /// - **Random**: In random order (debug) or arbitrary order (release)
    /// - **Sieve**: Following eviction order.
    ///
    /// # Returns
    /// An iterator that yields `(K, V)` pairs in eviction order, consuming the
    /// cache.
    ///
    /// # Examples
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert(1, "first");
    /// cache.insert(2, "second");
    /// cache.insert(3, "third");
    ///
    /// // Convert to iterator and collect all pairs
    /// let pairs: Vec<_> = cache.into_iter().collect();
    /// assert_eq!(pairs, [(1, "first"), (2, "second"), (3, "third")]);
    /// ```
    ///
    /// # Performance
    /// - Time complexity: O(n) where n is the number of cached items
    /// - Space complexity: O(n) for the iterator's internal state
    ///
    /// # Note
    /// This is a consuming operation - the cache cannot be used after calling
    /// `into_iter()`. Use `iter()` if you need to iterate without consuming the
    /// cache.
    fn into_iter(self) -> Self::IntoIter {
        PolicyType::into_iter(self.metadata, self.queue)
    }
}

impl<Key: Hash + Eq, Value, PolicyType: Policy<Value>> std::iter::FromIterator<(Key, Value)>
    for Cache<Key, Value, PolicyType>
{
    /// Creates a new cache from an iterator of key-value pairs.
    ///
    /// This method consumes the iterator and constructs a new cache, with a
    /// **capacity of at least 1** and at most the number of items in the
    /// iterator.
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Key, Value)>,
    {
        let mut queue: IndexMap<
            Key,
            <PolicyType::MetadataType as Metadata<Value>>::EntryType,
            RandomState,
        > = IndexMap::with_hasher(RandomState::default());
        let mut metadata = PolicyType::MetadataType::default();

        #[cfg(feature = "statistics")]
        let mut hits = 0;
        #[cfg(feature = "statistics")]
        let mut insertions = 0;
        #[cfg(feature = "statistics")]
        let mut misses = 0;

        for (key, value) in iter {
            match queue.entry(key) {
                indexmap::map::Entry::Occupied(mut o) => {
                    *o.get_mut().value_mut() = value;
                    PolicyType::touch_entry(o.index(), false, &mut metadata, &mut queue);

                    #[cfg(feature = "statistics")]
                    {
                        hits += 1;
                    }
                }
                indexmap::map::Entry::Vacant(v) => {
                    let index = v.index();
                    v.insert(<PolicyType::MetadataType as Metadata<Value>>::EntryType::new(value));
                    PolicyType::touch_entry(index, false, &mut metadata, &mut queue);

                    #[cfg(feature = "statistics")]
                    {
                        insertions += 1;
                        misses += 1;
                    }
                }
            }
        }

        let capacity = NonZeroUsize::new(queue.len().max(1)).unwrap();
        Self {
            #[cfg(feature = "statistics")]
            statistics: Statistics {
                hits,
                insertions,
                evictions: 0,
                len: queue.len(),
                capacity,
                misses,
            },
            queue,
            capacity,
            metadata,
        }
    }
}

impl<K: Hash + Eq, V, PolicyType: Policy<V>> Extend<(K, V)> for Cache<K, V, PolicyType> {
    /// Extends the cache with key-value pairs from an iterator.
    ///
    /// This method consumes the iterator and inserts each `(Key, Value)` pair
    /// into the cache. If the cache is at capacity, the policy determines which
    /// entry is evicted to make room for new entries.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator of `(Key, Value)` pairs to insert into the cache
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    /// cache.extend(vec![(1, "one".to_string()), (2, "two".to_string())]);
    /// assert_eq!(cache.len(), 2);
    /// ```
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    //! These are mostly just tests to make sure that if the stored k/v pairs
    //! implement various traits, the Cache also implements those traits.

    #[test]
    fn test_lru_cache_clone() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            LruPolicy,
        };

        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let cloned_cache = cache.clone();
        assert_eq!(cloned_cache.len(), 2);
        assert_eq!(cloned_cache.peek(&1), Some(&"one".to_string()));
        assert_eq!(cloned_cache.peek(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_lfu_cache_clone() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            LfuPolicy,
        };

        let mut cache = Cache::<i32, String, LfuPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let cloned_cache = cache.clone();
        assert_eq!(cloned_cache.len(), 2);
        assert_eq!(cloned_cache.peek(&1), Some(&"one".to_string()));
        assert_eq!(cloned_cache.peek(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_mru_cache_clone() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            MruPolicy,
        };
        let mut cache = Cache::<i32, String, MruPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        let cloned_cache = cache.clone();
        assert_eq!(cloned_cache.len(), 2);
        assert_eq!(cloned_cache.peek(&1), Some(&"one".to_string()));
        assert_eq!(cloned_cache.peek(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_fifo_cache_clone() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            FifoPolicy,
        };

        let mut cache = Cache::<i32, String, FifoPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let cloned_cache = cache.clone();
        assert_eq!(cloned_cache.len(), 2);
        assert_eq!(cloned_cache.peek(&1), Some(&"one".to_string()));
        assert_eq!(cloned_cache.peek(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_lifo_cache_clone() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            LifoPolicy,
        };

        let mut cache = Cache::<i32, String, LifoPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let cloned_cache = cache.clone();
        assert_eq!(cloned_cache.len(), 2);
        assert_eq!(cloned_cache.peek(&1), Some(&"one".to_string()));
        assert_eq!(cloned_cache.peek(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_random_cache_clone() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            RandomPolicy,
        };

        let mut cache = Cache::<i32, String, RandomPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let cloned_cache = cache.clone();
        assert_eq!(cloned_cache.len(), 2);
        assert_eq!(cloned_cache.peek(&1), Some(&"one".to_string()));
        assert_eq!(cloned_cache.peek(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_sieve_cache_clone() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            SievePolicy,
        };

        let mut cache = Cache::<i32, String, SievePolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let cloned_cache = cache.clone();
        assert_eq!(cloned_cache.len(), 2);
        assert_eq!(cloned_cache.peek(&1), Some(&"one".to_string()));
        assert_eq!(cloned_cache.peek(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_lfu_cache_debug() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            LfuPolicy,
        };

        let mut cache = Cache::<i32, String, LfuPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
        assert!(debug_str.contains("\"one\""));
        assert!(debug_str.contains("\"two\""));
    }

    #[test]
    fn test_lru_cache_debug() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            LruPolicy,
        };

        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
        assert!(debug_str.contains("\"one\""));
        assert!(debug_str.contains("\"two\""));
    }

    #[test]
    fn test_mru_cache_debug() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            MruPolicy,
        };

        let mut cache = Cache::<i32, String, MruPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
        assert!(debug_str.contains("\"one\""));
        assert!(debug_str.contains("\"two\""));
    }

    #[test]
    fn test_fifo_cache_debug() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            FifoPolicy,
        };

        let mut cache = Cache::<i32, String, FifoPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
        assert!(debug_str.contains("\"one\""));
        assert!(debug_str.contains("\"two\""));
    }

    #[test]
    fn test_lifo_cache_debug() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            LifoPolicy,
        };

        let mut cache = Cache::<i32, String, LifoPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
        assert!(debug_str.contains("\"one\""));
        assert!(debug_str.contains("\"two\""));
    }

    #[test]
    fn test_random_cache_debug() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            RandomPolicy,
        };

        let mut cache = Cache::<i32, String, RandomPolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
        assert!(debug_str.contains("\"one\""));
        assert!(debug_str.contains("\"two\""));
    }

    #[test]
    fn test_sieve_cache_debug() {
        use std::num::NonZeroUsize;

        use crate::{
            Cache,
            SievePolicy,
        };

        let mut cache = Cache::<i32, String, SievePolicy>::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
        assert!(debug_str.contains("\"one\""));
        assert!(debug_str.contains("\"two\""));
    }
}

#[cfg(all(test, feature = "statistics"))]
mod statistics_tests {
    use super::*;

    #[test]
    fn test_statistics_initialization() {
        let cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());
        let stats = cache.statistics();

        assert_eq!(stats.len, 0);
        assert_eq!(stats.capacity.get(), 3);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.insertions, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn test_statistics_insertions_and_misses() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let stats = cache.statistics();
        assert_eq!(stats.len, 3);
        assert_eq!(stats.insertions, 3);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.evictions, 0);
    }

    #[test]
    fn test_statistics_hits() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        // Test hits with get
        cache.get(&1);
        cache.get(&2);
        cache.get(&1);

        let stats = cache.statistics();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.insertions, 2);
    }

    #[test]
    fn test_statistics_misses_on_failed_get() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());

        // Miss on non-existent key
        cache.get(&999);
        cache.get(&888);

        let stats = cache.statistics();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.insertions, 1);
    }

    #[test]
    fn test_statistics_evictions() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let stats_before = cache.statistics();
        assert_eq!(stats_before.evictions, 0);

        // This should trigger an eviction
        cache.insert(3, "three".to_string());

        let stats_after = cache.statistics();
        assert_eq!(stats_after.evictions, 1);
        assert_eq!(stats_after.len, 2);
        assert_eq!(stats_after.insertions, 3);
    }

    #[test]
    fn test_statistics_with_get_or_insert_with() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        // Insert through get_or_insert_with
        cache.get_or_insert_with(1, |_| "one".to_string());
        cache.get_or_insert_with(2, |_| "two".to_string());

        let stats = cache.statistics();
        assert_eq!(stats.insertions, 2);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);

        // Hit through get_or_insert_with
        cache.get_or_insert_with(1, |_| "one_new".to_string());

        let stats = cache.statistics();
        assert_eq!(stats.insertions, 2);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_statistics_with_try_insert() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(2).unwrap());

        // Successful insertions
        assert!(cache.try_insert_or_update(1, "one".to_string()).is_ok());
        assert!(cache.try_insert_or_update(2, "two".to_string()).is_ok());

        let stats = cache.statistics();
        assert_eq!(stats.insertions, 2);
        assert_eq!(stats.misses, 2);

        // Failed insertion (key already exists)
        assert_eq!(
            cache.try_insert_or_update(1, "one_new".to_string()),
            Ok(&"one_new".to_string())
        );

        let stats = cache.statistics();
        assert_eq!(stats.insertions, 2); // Should not increment
        assert_eq!(stats.hits, 1); // Should increment for the failed try_insert
    }

    #[test]
    fn test_statistics_reset() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.get(&1);
        cache.get(&999); // miss

        let stats_before = cache.statistics();
        assert!(stats_before.hits > 0);
        assert!(stats_before.misses > 0);
        assert!(stats_before.insertions > 0);

        cache.reset_statistics();

        let stats_after = cache.statistics();
        assert_eq!(stats_after.hits, 0);
        assert_eq!(stats_after.misses, 0);
        assert_eq!(stats_after.insertions, 0);
        assert_eq!(stats_after.evictions, 0);
        assert_eq!(stats_after.len, 2); // len should match current cache state
        assert_eq!(stats_after.capacity.get(), 3);
    }

    #[test]
    fn test_statistics_with_remove() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let stats_before = cache.statistics();
        assert_eq!(stats_before.len, 3);

        cache.remove(&2);

        let stats_after = cache.statistics();
        assert_eq!(stats_after.len, 2);
        // Remove shouldn't affect other statistics
        assert_eq!(stats_after.insertions, stats_before.insertions);
        assert_eq!(stats_after.hits, stats_before.hits);
        assert_eq!(stats_after.misses, stats_before.misses);
    }

    #[test]
    fn test_statistics_with_clear() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.get(&1);

        let stats_before = cache.statistics();
        assert!(stats_before.len > 0);

        cache.clear();

        let stats_after = cache.statistics();
        assert_eq!(stats_after.len, 0);
        // Clear shouldn't affect historical statistics
        assert_eq!(stats_after.insertions, stats_before.insertions);
        assert_eq!(stats_after.hits, stats_before.hits);
        assert_eq!(stats_after.misses, stats_before.misses);
    }

    #[test]
    fn test_statistics_with_set_capacity() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let stats_before = cache.statistics();
        assert_eq!(stats_before.capacity.get(), 3);
        assert_eq!(stats_before.evictions, 0);

        // Set capacity to smaller size should trigger evictions
        let evicted = cache.set_capacity(NonZeroUsize::new(1).unwrap());
        assert_eq!(evicted.len(), 2); // Should evict 2 items

        let stats_after = cache.statistics();
        assert_eq!(stats_after.capacity.get(), 1);
        assert_eq!(stats_after.len, 1);
        assert_eq!(stats_after.evictions, 2); // Should evict 2 items
    }

    #[test]
    fn test_statistics_with_different_policies() {
        // Test with LFU policy
        let mut lfu_cache = Cache::<i32, String, LfuPolicy>::new(NonZeroUsize::new(2).unwrap());
        lfu_cache.insert(1, "one".to_string());
        lfu_cache.insert(2, "two".to_string());
        lfu_cache.get(&1); // Increase frequency of key 1
        lfu_cache.insert(3, "three".to_string()); // Should evict key 2 (least frequently used)

        let lfu_stats = lfu_cache.statistics();
        assert_eq!(lfu_stats.evictions, 1);
        assert_eq!(lfu_stats.hits, 1);
        assert_eq!(lfu_stats.insertions, 3);

        // Test with MRU policy
        let mut mru_cache = Cache::<i32, String, MruPolicy>::new(NonZeroUsize::new(2).unwrap());
        mru_cache.insert(1, "one".to_string());
        mru_cache.insert(2, "two".to_string());
        mru_cache.get(&1); // Make key 1 most recently used
        mru_cache.insert(3, "three".to_string()); // Should evict key 1 (most recently used)

        let mru_stats = mru_cache.statistics();
        assert_eq!(mru_stats.evictions, 1);
        assert_eq!(mru_stats.hits, 1);
        assert_eq!(mru_stats.insertions, 3);
    }

    #[test]
    fn test_statistics_clone() {
        let mut cache = Cache::<i32, String, LruPolicy>::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.get(&1);
        cache.get(&999); // miss

        let cloned_cache = cache.clone();
        let original_stats = cache.statistics();
        let cloned_stats = cloned_cache.statistics();

        // Statistics should be identical after clone
        assert_eq!(original_stats.len, cloned_stats.len);
        assert_eq!(original_stats.capacity, cloned_stats.capacity);
        assert_eq!(original_stats.hits, cloned_stats.hits);
        assert_eq!(original_stats.misses, cloned_stats.misses);
        assert_eq!(original_stats.insertions, cloned_stats.insertions);
        assert_eq!(original_stats.evictions, cloned_stats.evictions);
    }
}
