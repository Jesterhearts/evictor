#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
mod lfu;
mod lru;
mod mru;

use std::{
    fmt::Debug,
    hash::Hash,
    num::NonZeroUsize,
};

use indexmap::IndexMap;
pub use lfu::LfuPolicy;
pub use lru::LruPolicy;
pub use mru::MruPolicy;

use crate::private::Sealed;

#[cfg(not(feature = "ahash"))]
type RandomState = std::hash::RandomState;
#[cfg(feature = "ahash")]
type RandomState = ahash::RandomState;

/// Private module to implement the sealed trait pattern.
mod private {
    /// Sealed trait to prevent external trait implementation.
    pub trait Sealed {}
}

/// Trait for wrapping values in cache entries.
///
/// This trait allows policies to associate additional metadata with cached
/// values while providing uniform access to the underlying data.
#[doc(hidden)]
pub trait EntryValue<T>: Sealed {
    /// Creates a new entry containing the given value.
    fn new(value: T) -> Self;

    /// Prepares the entry for reinsertion into the cache after being
    /// temporarily removed and examined.
    fn prepare_for_reinsert(&mut self);

    /// Converts the entry into its contained value.
    fn into_value(self) -> T;

    /// Returns a reference to the contained value.
    fn value(&self) -> &T;

    /// Returns a mutable reference to the contained value.
    fn value_mut(&mut self) -> &mut T;
}

/// Trait for cache metadata that tracks eviction state.
///
/// This trait provides policy-specific metadata storage and the ability
/// to identify which entry should be evicted next.
#[doc(hidden)]
pub trait Metadata: Default + Sealed {
    /// Returns the index of the entry that should be evicted next.
    fn tail_index(&self) -> usize;
}

/// Trait defining cache eviction policies.
///
/// This trait encapsulates the behavior of different cache eviction strategies
/// such as LRU, MRU, and LFU. Each policy defines how entries are prioritized
/// and which entry should be evicted when the cache is full.
#[doc(hidden)]
pub trait Policy<T>: Sealed {
    /// The entry type used by this policy to wrap cached values.
    type EntryType: EntryValue<T>;

    /// The metadata type used by this policy to track eviction state.
    type MetadataType: Metadata;

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
    fn touch_entry(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<impl Hash + Eq, Self::EntryType, RandomState>,
    ) -> usize;

    /// Returns an iterator over the removed pairs in insertion order.
    /// In most cases, this will be the same as the order of removal.
    /// In certain cases (e.g. MRU), the order may differ.
    fn iter_removed_pairs_in_insertion_order<K>(
        pairs: Vec<(K, Self::EntryType)>,
    ) -> impl Iterator<Item = (K, Self::EntryType)> {
        pairs.into_iter()
    }

    /// Removes the entry at the specified index and returns the index of the
    /// entry which replaced it.
    ///
    /// This method handles the removal of an entry and updates the policy's
    /// internal state to maintain consistency.
    ///
    /// # Parameters
    /// - `index`: The index of the entry to remove
    /// - `metadata`: Mutable reference to policy metadata
    /// - `queue`: Mutable reference to the cache's storage
    ///
    /// # Returns
    /// A tuple containing:
    /// - The index that replaced the removed entry (for index map consistency)
    /// - The removed key-value pair, or None if the index was invalid
    fn swap_remove_entry<K: Hash + Eq>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<K, Self::EntryType, RandomState>,
    ) -> (usize, Option<(K, Self::EntryType)>);
}

/// A least-recently-used (LRU) cache.
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
/// assert!(cache.contains_key(&1)); // Recently used, kept
/// assert!(!cache.contains_key(&2)); // Least recently used, evicted
/// assert!(cache.contains_key(&3));
/// assert!(cache.contains_key(&4));
/// ```
pub type Lru<Key, Value> = Cache<Key, Value, LruPolicy>;

/// A most-recently-used (MRU) cache.
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
/// assert!(!cache.contains_key(&1)); // Most recently used, evicted
/// assert!(cache.contains_key(&2));
/// assert!(cache.contains_key(&3));
/// assert!(cache.contains_key(&4));
/// ```
pub type Mru<Key, Value> = Cache<Key, Value, MruPolicy>;

/// A least-frequently-used (LFU) cache.
///
/// The cache maintains at most `capacity` entries. When inserting into a full
/// cache, the least frequently used entry is automatically evicted. All
/// operations that access values (get, insert, get_or_insert_with) increment
/// the entry's frequency counter. Operations that only read without accessing
/// (peek, contains_key, tail) do not affect frequency tracking.
///
/// Unlike LRU which tracks recency of access, LFU tracks frequency of access,
/// making it useful for scenarios where you want to keep items that are
/// accessed repeatedly, regardless of when they were last accessed.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(log n) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop (least-frequently-used): O(log n)
/// - Clear: O(1)
///
/// # Space Complexity
/// - O(`capacity`) memory usage
/// - Pre-allocates space for `capacity` entries
///
/// # Frequency Tracking
///
/// Each entry maintains a frequency counter that starts at 0 and increments
/// on every access. When eviction is needed, the entry with the lowest
/// frequency is removed. If multiple entries have the same frequency, the
/// implementation will choose one based on internal heap ordering.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Lfu;
///
/// let mut cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());
///
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// assert_eq!(cache.len(), 3);
///
/// // Access key 1 multiple times to increase its frequency
/// cache.get(&1);
/// cache.get(&1);
/// cache.get(&1);
///
/// // Access key 2 once
/// cache.get(&2);
///
/// // Key 3 has frequency 0 (never accessed after insertion)
/// // Insert new item - this will evict the least frequently used (key 3)
/// cache.insert(4, "four".to_string());
/// assert!(!cache.contains_key(&3)); // Evicted (frequency 0)
/// assert!(cache.contains_key(&1)); // Kept (frequency 3)
/// assert!(cache.contains_key(&2)); // Kept (frequency 1)
/// assert!(cache.contains_key(&4)); // Newly inserted
/// ```
///
/// ## Comparison with LRU
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::{
///     Lfu,
///     Lru,
/// };
///
/// let mut lfu = Lfu::<i32, String>::new(NonZeroUsize::new(2).unwrap());
/// let mut lru = Lru::<i32, String>::new(NonZeroUsize::new(2).unwrap());
///
/// // Same initial setup
/// lfu.insert(1, "one".to_string());
/// lfu.insert(2, "two".to_string());
/// lru.insert(1, "one".to_string());
/// lru.insert(2, "two".to_string());
///
/// // Access key 1 multiple times
/// lfu.get(&1);
/// lfu.get(&1);
/// lru.get(&1); // Only matters that it was accessed recently
///
/// // Insert new item causing eviction
/// lfu.insert(3, "three".to_string());
/// lru.insert(3, "three".to_string());
///
/// // LFU keeps frequently accessed item (key 1), evicts key 2
/// assert!(lfu.contains_key(&1));
/// assert!(!lfu.contains_key(&2));
///
/// // LRU keeps recently accessed item (key 1), evicts key 2
/// assert!(lru.contains_key(&1));
/// assert!(!lru.contains_key(&2));
///
/// // But with different access patterns, results differ:
/// let mut lfu2 = Lfu::<i32, String>::new(NonZeroUsize::new(2).unwrap());
/// let mut lru2 = Lru::<i32, String>::new(NonZeroUsize::new(2).unwrap());
///
/// lfu2.insert(1, "one".to_string());
/// lfu2.insert(2, "two".to_string());
/// lru2.insert(1, "one".to_string());
/// lru2.insert(2, "two".to_string());
///
/// // Access key 1 frequently, then key 2 once (recently)
/// for _ in 0..5 {
///     lfu2.get(&1);
/// }
/// for _ in 0..5 {
///     lru2.get(&1);
/// }
/// lfu2.get(&2);
/// lru2.get(&2);
///
/// lfu2.insert(3, "three".to_string());
/// lru2.insert(3, "three".to_string());
///
/// // LFU keeps the frequently used key 1, evicts recently used key 2
/// assert!(lfu2.contains_key(&1));
/// assert!(!lfu2.contains_key(&2));
///
/// // LRU keeps the recently used key 2, evicts key 1
/// assert!(!lru2.contains_key(&1));
/// assert!(lru2.contains_key(&2));
/// ```
pub type Lfu<Key, Value> = Cache<Key, Value, LfuPolicy>;

/// A generic cache implementation with configurable eviction policies.
///
/// `Cache` is the underlying generic structure that powers LRU, MRU, and LFU
/// caches. The eviction behavior is determined by the `PolicyType` parameter,
/// which must implement the `Policy` trait.
///
/// For most use cases, you should use the type aliases [`Lru`], [`Mru`], or
/// [`Lfu`] instead of using `Cache` directly.
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
#[derive(Debug, Clone)]
pub struct Cache<Key, Value, PolicyType: Policy<Value>> {
    queue: IndexMap<Key, PolicyType::EntryType, RandomState>,
    capacity: NonZeroUsize,
    metadata: PolicyType::MetadataType,
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
    /// entry as recently accessed.
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
    ///
    /// // Peek doesn't affect eviction order
    /// assert_eq!(cache.peek(&1), Some(&"one".to_string()));
    /// assert_eq!(cache.peek(&2), None);
    /// ```
    pub fn peek(&self, key: &Key) -> Option<&Value> {
        self.queue.get(key).map(|entry| entry.value())
    }

    /// Returns a reference to the entry that would be evicted next.
    ///
    /// This method provides access to the entry at the "tail" of the eviction
    /// order without affecting the cache state. The specific entry returned
    /// depends on the policy:
    /// - LRU: Returns the least recently used entry
    /// - MRU: Returns the most recently used entry
    /// - LFU: Returns the least frequently used entry
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
    /// // In LRU, tail returns the least recently used
    /// let (key, value) = cache.tail().unwrap();
    /// assert_eq!(key, &1);
    /// assert_eq!(value, &"one".to_string());
    /// ```
    pub fn tail(&self) -> Option<(&Key, &Value)> {
        self.queue
            .get_index(self.metadata.tail_index())
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
    /// it as recently accessed. If the key doesn't exist, calls the provided
    /// function to create a new value, inserts it, and returns a reference to
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
    /// it as recently accessed. If the key doesn't exist, calls the provided
    /// function to create a new value, inserts it, and returns a mutable
    /// reference to it.
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
        let mut index = match self.queue.entry(key) {
            indexmap::map::Entry::Occupied(o) => o.index(),
            indexmap::map::Entry::Vacant(v) => {
                let mut index = v.index();
                let e = PolicyType::EntryType::new(or_insert(v.key()));
                v.insert(e);

                // Our previous len was at capacity, but we just inserted a new entry.
                // So we need to remove the oldest entry.
                if len == self.capacity.get() {
                    index = PolicyType::swap_remove_entry(
                        self.metadata.tail_index(),
                        &mut self.metadata,
                        &mut self.queue,
                    )
                    .0;
                }

                index
            }
        };

        index = PolicyType::touch_entry(index, &mut self.metadata, &mut self.queue);
        self.queue[index].value_mut()
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// If the key already exists, its value is updated and the entry is marked
    /// as recently accessed. If the key is new and the cache is at capacity,
    /// the policy determines which entry is evicted to make room.
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
    /// assert!(!cache.contains_key(&1));
    /// assert!(cache.contains_key(&3));
    /// ```
    pub fn insert(&mut self, key: Key, value: Value) -> &Value {
        self.insert_mut(key, value)
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// This is the mutable version of [`insert()`](Self::insert). If the key
    /// already exists, its value is updated and the entry is marked as recently
    /// accessed. If the key is new and the cache is at capacity, the policy
    /// determines which entry is evicted to make room.
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
    /// assert_eq!(cache.len(), 2);
    /// ```
    pub fn insert_mut(&mut self, key: Key, value: Value) -> &mut Value {
        let len = self.queue.len();
        let mut index = match self.queue.entry(key) {
            indexmap::map::Entry::Occupied(mut o) => {
                *o.get_mut().value_mut() = value;
                o.index()
            }
            indexmap::map::Entry::Vacant(v) => {
                let mut index = v.index();
                v.insert(PolicyType::EntryType::new(value));

                // Our previous len was at capacity, but we just inserted a new entry.
                // So we need to remove the oldest entry.
                if len == self.capacity.get() {
                    index = PolicyType::swap_remove_entry(
                        self.metadata.tail_index(),
                        &mut self.metadata,
                        &mut self.queue,
                    )
                    .0;
                }

                index
            }
        };

        index = PolicyType::touch_entry(index, &mut self.metadata, &mut self.queue);
        self.queue[index].value_mut()
    }

    /// Gets a value from the cache, marking it as recently used.
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
    /// ```
    pub fn get(&mut self, key: &Key) -> Option<&Value> {
        self.get_mut(key).map(|v| &*v)
    }

    /// Gets a value from the cache, marking it as recently used.
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
    /// ```
    pub fn get_mut(&mut self, key: &Key) -> Option<&mut Value> {
        if let Some(index) = self.queue.get_index_of(key) {
            let index = PolicyType::touch_entry(index, &mut self.metadata, &mut self.queue);
            Some(self.queue[index].value_mut())
        } else {
            None
        }
    }

    /// Removes and returns the entry that would be evicted next.
    ///
    /// This method removes and returns the entry at the "tail" of the eviction
    /// order. The specific entry removed depends on the policy:
    /// - LRU: Removes the least recently used entry
    /// - MRU: Removes the most recently used entry
    /// - LFU: Removes the least frequently used entry
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
    /// assert_eq!(cache.len(), 1);
    /// ```
    pub fn pop(&mut self) -> Option<(Key, Value)> {
        PolicyType::swap_remove_entry(
            self.metadata.tail_index(),
            &mut self.metadata,
            &mut self.queue,
        )
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
    /// assert_eq!(cache.len(), 1);
    /// ```
    pub fn remove(&mut self, key: &Key) -> Option<Value> {
        if let Some(index) = self.queue.get_index_of(key) {
            PolicyType::swap_remove_entry(index, &mut self.metadata, &mut self.queue)
                .1
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

    /// Extends the cache with key-value pairs from an iterator.
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
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    ///
    /// let items = vec![(3, "three".to_string()), (4, "four".to_string())];
    /// cache.extend(items);
    ///
    /// assert_eq!(cache.len(), 3);
    /// assert_eq!(cache.peek(&2), Some(&"two".to_string()));
    /// assert_eq!(cache.peek(&3), Some(&"three".to_string()));
    /// assert_eq!(cache.peek(&4), Some(&"four".to_string()));
    /// ```
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (Key, Value)>,
    {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }

    /// TODO
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&Key, &mut Value) -> bool,
    {
        let mut kvs = Vec::with_capacity(self.queue.len());
        while let Some((key, mut entry)) = PolicyType::swap_remove_entry(
            self.metadata.tail_index(),
            &mut self.metadata,
            &mut self.queue,
        )
        .1
        {
            if f(&key, entry.value_mut()) {
                entry.prepare_for_reinsert();
                kvs.push((key, entry));
            }
        }

        self.metadata = PolicyType::MetadataType::default();
        for (key, entry) in PolicyType::iter_removed_pairs_in_insertion_order(kvs) {
            self.queue.insert(key, entry);
            PolicyType::touch_entry(self.queue.len() - 1, &mut self.metadata, &mut self.queue);
        }
    }

    /// Shrinks the internal storage to fit the current number of entries.
    pub fn shrink_to_fit(&mut self) {
        self.queue.shrink_to_fit();
    }
}

impl<Key: Hash + Eq, Value, PolicyType: Policy<Value>> std::iter::FromIterator<(Key, Value)>
    for Cache<Key, Value, PolicyType>
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Key, Value)>,
    {
        let mut queue: IndexMap<Key, PolicyType::EntryType, ahash::RandomState> =
            IndexMap::with_hasher(RandomState::default());
        let mut metadata = PolicyType::MetadataType::default();

        for (key, value) in iter {
            match queue.entry(key) {
                indexmap::map::Entry::Occupied(mut o) => {
                    *o.get_mut().value_mut() = value;
                    PolicyType::touch_entry(o.index(), &mut metadata, &mut queue);
                }
                indexmap::map::Entry::Vacant(v) => {
                    let index = v.index();
                    v.insert(PolicyType::EntryType::new(value));
                    PolicyType::touch_entry(index, &mut metadata, &mut queue);
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

#[cfg(test)]
mod tests {
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
    fn test_heap_lfu_cache_clone() {
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
}
