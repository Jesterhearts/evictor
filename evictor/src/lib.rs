#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]
#![cfg_attr(all(doc, ENABLE_DOC_AUTO_CFG), feature(doc_auto_cfg))]

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
pub type Lru<Key, Value> = Cache<Key, Value, LruPolicy>;

/// See [`Lru`].
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
pub type Mru<Key, Value> = Cache<Key, Value, MruPolicy>;

/// See [`Mru`].
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
pub type Lfu<Key, Value> = Cache<Key, Value, LfuPolicy>;

/// See [`Lfu`].
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
pub type Fifo<Key, Value> = Cache<Key, Value, FifoPolicy>;

/// See [`Fifo`].
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
pub type Lifo<Key, Value> = Cache<Key, Value, LifoPolicy>;

/// See [`Lifo`].
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
/// Sieve is a simple and efficient eviction policy that provides performance
/// comparable to Lru while being significantly easier to implement and
/// maintain. It uses a "visited" bit per entry and a hand pointer that scans
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
pub type Sieve<Key, Value> = Cache<Key, Value, SievePolicy>;

/// See [`Sieve`].
pub type SIEVE<Key, Value> = Sieve<Key, Value>;

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
        }
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
                self.queue[index].value_mut()
            }
            indexmap::map::Entry::Vacant(v) => {
                let index = v.index();
                let e = <PolicyType::MetadataType as Metadata<Value>>::EntryType::new(or_insert(
                    v.key(),
                ));
                v.insert(e);
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

    /// Inserts a key-value pair into the cache.
    ///
    /// If the key already exists, its value is updated and the entry is marked
    /// as touched. If the key is new and the cache is at capacity, the policy
    /// determines which entry is evicted to make room.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to associate with the key
    ///
    /// # Returns
    ///
    /// An immutable reference to the inserted value.
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
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    /// assert_eq!(cache.len(), 2);
    ///
    /// // This will evict the least recently used entry (key 1)
    /// cache.insert(3, "three".to_string());
    ///
    /// assert_eq!(
    ///     cache.into_iter().collect::<Vec<_>>(),
    ///     [(2, "two".to_string()), (3, "three".to_string())]
    /// );
    /// ```
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

                self.queue[index].value_mut()
            }
        }
    }

    /// Gets a value from the cache, marking it as touched.
    ///
    /// If the key exists, returns a reference to the value and updates the
    /// entry's position in the eviction order according to the policy.
    /// If the key doesn't exist, returns `None` and the cache is unchanged.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// * `Some(&Value)` if the key exists
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
    /// // Get and mark as recently used
    /// assert_eq!(cache.get(&1), Some(&"one".to_string()));
    /// assert_eq!(cache.get(&3), None);
    /// assert_eq!(
    ///     cache.into_iter().collect::<Vec<_>>(),
    ///     vec![(2, "two".to_string()), (1, "one".to_string())]
    /// );
    /// ```
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
            Some(self.queue[index].value_mut())
        } else {
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
            .map(|(key, entry)| (key, entry.into_value()))
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
    /// This value is set when the cache is created and doesn't change during
    /// the cache's lifetime.
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
    /// Iterates in **eviction order assuming no modifications**:
    /// - Starts from the current hand position (next eviction candidate)
    /// - Simulates the eviction scanning process: skips visited entries once,
    ///   then includes them in order
    /// - Uses additional state tracking and map lookups to maintain this
    ///   logical ordering
    /// - **Higher overhead** than other policies due to eviction simulation
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

        for (key, value) in iter {
            match queue.entry(key) {
                indexmap::map::Entry::Occupied(mut o) => {
                    *o.get_mut().value_mut() = value;
                    PolicyType::touch_entry(o.index(), false, &mut metadata, &mut queue);
                }
                indexmap::map::Entry::Vacant(v) => {
                    let index = v.index();
                    v.insert(<PolicyType::MetadataType as Metadata<Value>>::EntryType::new(value));
                    PolicyType::touch_entry(index, false, &mut metadata, &mut queue);
                }
            }
        }

        let capacity = NonZeroUsize::new(queue.len().max(1)).unwrap();
        Self {
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
