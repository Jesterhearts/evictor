#![doc = include_str!("../README.md")]
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

mod private {
    use crate::{
        Entry,
        lfu::LfuPolicy,
        lru::LruPolicy,
        mru::MruPolicy,
    };
    pub trait Sealed {}
    impl Sealed for LruPolicy {}
    impl Sealed for MruPolicy {}
    impl Sealed for LfuPolicy {}

    impl<T> Sealed for Entry<T> {}
}

#[derive(Debug)]
struct Entry<Value> {
    priority: u64,
    value: Value,
}

pub trait EntryValue: Sealed {
    fn priority(&self) -> u64;
    fn priority_mut(&mut self) -> &mut u64;
}

impl<Value> EntryValue for Entry<Value> {
    fn priority(&self) -> u64 {
        self.priority
    }

    fn priority_mut(&mut self) -> &mut u64 {
        &mut self.priority
    }
}

/// Trait defining cache eviction policies.
///
/// This trait provides the core interface for implementing different cache
/// eviction strategies. Each policy determines how entries are prioritized and
/// which entries should be evicted when the cache reaches capacity.
///
/// # Implementation Overview
///
/// The policy controls cache behavior through a priority-based system where
/// each entry has a numeric priority value. The cache maintains these entries
/// in a heap structure, with the entry at index 0 always being the next
/// candidate for eviction.
///
/// ## Priority Systems
///
/// Different policies use different priority assignment strategies:
///
/// - **LRU (Least Recently Used)**: Uses descending priorities (u64::MAX →
///   u64::MIN) where lower values indicate more recent access
/// - **MRU (Most Recently Used)**: Uses ascending priorities (u64::MIN →
///   u64::MAX) where higher values indicate more recent access
/// - **LFU (Least Frequently Used)**: Uses frequency counters starting at 0,
///   incremented on each access
///
/// ## Heap Invariants
///
/// The implementation must maintain heap properties:
/// - **LRU**: Max-heap where parent priority ≥ child priority
/// - **MRU**: Max-heap where parent priority ≥ child priority
/// - **LFU**: Min-heap where parent priority ≤ child priority
///
/// # Required Methods
///
/// Implementers must provide:
/// 1. **Priority bounds**: `start_value()` and `end_value()`
/// 2. **Priority assignment**: `assign_update_next_value()`
/// 3. **Heap maintenance**: `heap_bubble()`
/// 4. **Overflow handling**: `re_index()`
///
/// # Sealed Trait
///
/// This trait is sealed and cannot be implemented outside of this crate. The
/// available implementations are [`LruPolicy`], [`MruPolicy`], and
/// [`LfuPolicy`].
///
/// [`LruPolicy`]: crate::LruPolicy
/// [`MruPolicy`]: crate::MruPolicy
/// [`LfuPolicy`]: crate::LfuPolicy
pub trait Policy: Sealed {
    /// Returns the initial priority value for new cache entries.
    ///
    /// This value is used when the cache is first created and represents
    /// the starting point of the priority range. Different policies use
    /// different starting values based on their priority assignment strategy.
    ///
    /// # Policy-Specific Values
    ///
    /// - **LRU**: `u64::MAX`
    /// - **MRU**: `u64::MIN`
    /// - **LFU**: `0` (frequency counter starts at zero)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use evictor::{
    ///     LfuPolicy,
    ///     LruPolicy,
    ///     MruPolicy,
    ///     Policy,
    /// };
    ///
    /// assert_eq!(LruPolicy::start_value(), u64::MAX);
    /// assert_eq!(MruPolicy::start_value(), u64::MIN);
    /// assert_eq!(LfuPolicy::start_value(), 0);
    /// ```
    fn start_value() -> u64;

    /// Returns the sentinel value indicating priority range exhaustion.
    ///
    /// When the cache's next priority value reaches this sentinel, the cache
    /// must perform re-indexing to reset the priority range. This prevents
    /// priority overflow and maintains proper cache ordering.
    ///
    /// # Policy-Specific Values
    ///
    /// - **LRU**: `u64::MIN` (when countdown reaches minimum)
    /// - **MRU**: `u64::MAX` (when count-up reaches maximum)
    /// - **LFU**: `1` (sentinel value, re-indexing not supported)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use evictor::{
    ///     LfuPolicy,
    ///     LruPolicy,
    ///     MruPolicy,
    ///     Policy,
    /// };
    ///
    /// assert_eq!(LruPolicy::end_value(), u64::MIN);
    /// assert_eq!(MruPolicy::end_value(), u64::MAX);
    /// assert_eq!(LfuPolicy::end_value(), 1);
    /// ```
    fn end_value() -> u64;

    /// Updates an entry's priority and advances the global priority counter.
    ///
    /// This method is called whenever an entry is accessed or inserted. It
    /// must:
    /// 1. Assign the current priority to the entry
    ///
    /// It **may** also modify the global priority counter (`cache_next`) if
    /// required for tracking invariants.
    ///
    /// The exact behavior depends on the policy's priority strategy.
    ///
    /// # Parameters
    ///
    /// * `cache_next` - Mutable reference to the cache's global priority
    ///   counter
    /// * `entry` - Mutable reference to the cache entry being updated
    ///
    /// # Policy Behavior
    ///
    /// - **LRU**: Assigns current counter value, then decrements counter
    /// - **MRU**: Assigns current counter value, then increments counter
    /// - **LFU**: Increments entry's frequency, ignores global counter
    fn assign_update_next_value(cache_next: &mut u64, entry: &mut impl EntryValue);

    /// Maintains heap ordering after priority changes.
    ///
    /// This method restores the heap invariant after an entry's priority has
    /// been modified. It may need to bubble the entry up or down in the
    /// heap depending on the new priority value and the policy's heap type
    /// (min-heap vs max-heap).
    ///
    /// # Parameters
    ///
    /// * `index` - The index of the entry that may be out of place
    /// * `queue` - Mutable reference to the cache's internal storage
    ///
    /// # Returns
    ///
    /// The final index where the entry settled after heap reordering.
    ///
    /// # Implementation Requirements
    ///
    /// - Must maintain the appropriate heap property for the policy
    /// - Should be efficient (typically O(log n) complexity)
    /// - Must handle edge cases (empty queue, single element, etc.)
    ///
    /// # Heap Properties by Policy
    ///
    /// - **LRU**: Max-heap (parent ≥ children) - recent items bubble down
    /// - **MRU**: Max-heap (parent ≥ children) - recent items bubble up
    /// - **LFU**: Min-heap (parent ≤ children) - frequent items bubble down
    fn heap_bubble<T: EntryValue>(
        index: usize,
        queue: &mut IndexMap<impl Hash + Eq, T, RandomState>,
    ) -> usize;

    /// Reassigns priorities when the global counter reaches overflow.
    ///
    /// This method is called when `cache_next` equals `end_value()`, indicating
    /// that the priority range has been exhausted. The implementation must
    /// reassign priorities to all entries to prevent overflow while maintaining
    /// relative ordering.
    ///
    /// # Parameters
    ///
    /// * `queue` - Mutable reference to the cache's internal storage
    /// * `next_priority` - Mutable reference to reset the global priority
    ///   counter
    /// * `index` - Starting index for recursive re-indexing (typically 0)
    ///
    /// # Implementation Requirements
    ///
    /// - Must preserve relative priority ordering
    /// - Should reset `next_priority` to a valid starting value
    /// - Must handle empty queues gracefully
    /// - May use recursive tree traversal for efficiency
    ///
    /// # Policy Support
    ///
    /// - **LRU**: Supported - resets priorities maintaining recency order
    /// - **MRU**: Supported - resets priorities maintaining recency order
    /// - **LFU**: Not supported - panics because frequencies can't be reset.
    ///   This should be unreachable during normal usage.
    fn re_index<T: EntryValue>(
        queue: &mut IndexMap<impl Hash + Eq, T, RandomState>,
        next_priority: &mut u64,
        index: usize,
    );
}

/// A least-recently-used (LRU) cache.
///
/// The cache maintains at most `capacity` entries. When inserting into a full
/// cache, the least recently used entry is automatically evicted. All
/// operations that access values (get, insert, get_or_insert_with) update the
/// entry's position in the LRU order. Operations that only read without
/// accessing (peek, contains_key, tail) do not affect LRU ordering.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(log n) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop (oldest): O(log n)
/// - Clear: O(1)
///
/// # Space Complexity
/// - O(`capacity`) memory usage
/// - Pre-allocates space for `capacity` entries
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Lru;
///
/// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
///
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// assert_eq!(cache.len(), 3);
///
/// cache.get(&1);
///
/// cache.insert(4, "four".to_string());
/// assert!(!cache.contains_key(&2));
///
/// cache.peek(&3);
/// let (oldest_key, _) = cache.tail().unwrap();
/// assert_eq!(oldest_key, &3);
/// ```
pub type Lru<Key, Value> = Cache<Key, Value, LruPolicy>;

/// A most-recently-used (MRU) cache.
///
/// The cache maintains at most `capacity` entries. When inserting into a full
/// cache, the most recently used entry is automatically evicted. All
/// operations that access values (get, insert, get_or_insert_with) update the
/// entry's position in the MRU order. Operations that only read without
/// accessing (peek, contains_key, tail) do not affect MRU ordering.
///
/// This is the opposite of LRU behavior - instead of keeping frequently used
/// items, MRU evicts the items that were just accessed, making it useful for
/// scenarios where you want to prioritize older, less-accessed items.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(log n) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case
/// - Pop (youngest): O(log n)
/// - Clear: O(1)
///
/// # Space Complexity
/// - O(`capacity`) memory usage
/// - Pre-allocates space for `capacity` entries
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use evictor::Mru;
///
/// let mut cache = Mru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
///
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// assert_eq!(cache.len(), 3);
///
/// // Access key 1, making it most recently used
/// cache.get(&1);
///
/// // Insert new item - this will evict the most recently used (key 1)
/// cache.insert(4, "four".to_string());
/// assert!(!cache.contains_key(&1));
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
/// which must implement the [`Policy`] trait.
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
///   [`Policy`].
///
/// # Internal Structure
///
/// The cache maintains a priority system where each entry has a numeric
/// priority that determines its position in the eviction order. The policy
/// determines how priorities are assigned and updated.
///
/// # Heap Invariant
///
/// The cache maintains a heap-like structure where:
/// - For LRU: Lower priority values indicate more recent access
/// - For MRU: Higher priority values indicate more recent access
/// - For LFU: Higher priority values indicate more frequent access
/// - The entry at index 0 is always the next candidate for eviction
///
/// # Memory Management
///
/// - Pre-allocates space for `capacity` entries to minimize reallocations
/// - Automatically evicts entries when capacity is exceeded
/// - Supports shrinking to reduce memory usage when needed
///
/// # Examples
///
/// ## Using with Custom Policy
///
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::{
///     Cache,
///     LruPolicy,
/// };
///
/// // Explicitly using Cache with LruPolicy (equivalent to Lru<i32, String>)
/// let mut cache: Cache<i32, String, LruPolicy> = Cache::new(NonZeroUsize::new(3).unwrap());
///
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// assert_eq!(cache.len(), 2);
/// ```
///
/// ## Type Alias Usage (Recommended)
///
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::Lru;
///
/// // Preferred way using type alias
/// let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
/// ```
#[derive(Debug)]
pub struct Cache<Key, Value, PolicyType> {
    queue: IndexMap<Key, Entry<Value>, RandomState>,
    capacity: NonZeroUsize,
    next_priority: u64,
    _policy: std::marker::PhantomData<PolicyType>,
}

impl<Key: Hash + Eq, Value, PolicyType: Policy> Cache<Key, Value, PolicyType> {
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
    /// use evictor::{
    ///     Cache,
    ///     LruPolicy,
    /// };
    ///
    /// let cache: Cache<i32, String, LruPolicy> = Cache::new(NonZeroUsize::new(100).unwrap());
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            queue: IndexMap::with_capacity_and_hasher(capacity.get(), RandomState::default()),
            capacity,
            next_priority: PolicyType::start_value(),
            _policy: std::marker::PhantomData,
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
        self.next_priority = PolicyType::start_value();
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
        self.queue.get(key).map(|entry| &entry.value)
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
        self.queue.first().map(|(key, entry)| (key, &entry.value))
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
        if self.next_priority == PolicyType::end_value() {
            PolicyType::re_index(&mut self.queue, &mut self.next_priority, 0);
        }

        let len = self.queue.len();
        let index = match self.queue.entry(key) {
            indexmap::map::Entry::Occupied(o) => o.index(),
            indexmap::map::Entry::Vacant(v) => {
                let mut index = v.index();
                let e = Entry {
                    priority: self.next_priority,
                    value: or_insert(v.key()),
                };
                v.insert(e);

                // Our previous len was at capacity, but we just inserted a new entry.
                // So we need to remove the oldest entry.
                if len == self.capacity.get() {
                    debug_assert_eq!(index, len);
                    self.queue.swap_remove_index(0);
                    index = 0;
                }

                index
            }
        };

        PolicyType::assign_update_next_value(&mut self.next_priority, &mut self.queue[index]);
        let index = PolicyType::heap_bubble(index, &mut self.queue);
        &mut self.queue[index].value
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
        if self.next_priority == PolicyType::end_value() {
            PolicyType::re_index(&mut self.queue, &mut self.next_priority, 0);
        }

        let mut index = self
            .queue
            .insert_full(
                key,
                Entry {
                    priority: self.next_priority,
                    value,
                },
            )
            .0;

        PolicyType::assign_update_next_value(&mut self.next_priority, &mut self.queue[index]);
        if self.queue.len() > self.capacity.get() {
            debug_assert_eq!(index, self.queue.len() - 1);
            self.queue.swap_remove_index(0);
            index = 0;
        }

        let index = PolicyType::heap_bubble(index, &mut self.queue);
        &mut self.queue[index].value
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
        if self.next_priority == PolicyType::end_value() {
            PolicyType::re_index(&mut self.queue, &mut self.next_priority, 0);
        }

        if let Some(mut index) = self.queue.get_index_of(key) {
            PolicyType::assign_update_next_value(&mut self.next_priority, &mut self.queue[index]);
            index = PolicyType::heap_bubble(index, &mut self.queue);

            Some(&mut self.queue[index].value)
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
        self.pop_internal().map(|(key, entry)| (key, entry.value))
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
            let entry = self.queue.swap_remove_index(index);
            if index < self.queue.len() {
                PolicyType::heap_bubble(index, &mut self.queue);
            }
            entry.map(|(_, entry)| entry.value)
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

    /// Retains only the entries for which the predicate returns true.
    pub fn retain(&mut self, mut f: impl FnMut(&Key, &mut Value) -> bool) {
        self.queue.retain(|key, entry| f(key, &mut entry.value));
        self.heapify();
    }

    /// Extends the cache with key-value pairs from an iterator.
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (Key, Value)>,
    {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }

    /// Shrinks the internal storage to fit the current number of entries.
    pub fn shrink_to_fit(&mut self) {
        self.queue.shrink_to_fit();
    }

    fn pop_internal(&mut self) -> Option<(Key, Entry<Value>)> {
        if self.queue.is_empty() {
            return None;
        }

        let result = self.queue.swap_remove_index(0);
        PolicyType::heap_bubble(0, &mut self.queue);
        result
    }

    fn heapify(&mut self) {
        if self.queue.is_empty() {
            return;
        }

        for i in (0..self.queue.len() / 2).rev() {
            PolicyType::heap_bubble(i, &mut self.queue);
        }
    }
}

impl<Key: Hash + Eq, Value, PolicyType: Policy> std::iter::FromIterator<(Key, Value)>
    for Cache<Key, Value, PolicyType>
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Key, Value)>,
    {
        let mut queue: IndexMap<Key, Entry<Value>, ahash::RandomState> =
            IndexMap::with_hasher(RandomState::default());
        let mut priority = PolicyType::start_value();

        for (key, value) in iter {
            match queue.entry(key) {
                indexmap::map::Entry::Occupied(mut o) => {
                    o.get_mut().value = value;
                    PolicyType::assign_update_next_value(&mut priority, o.get_mut());
                }
                indexmap::map::Entry::Vacant(v) => {
                    let entry = v.insert(Entry { priority, value });
                    PolicyType::assign_update_next_value(&mut priority, entry);
                }
            }
        }

        let capacity = NonZeroUsize::new(queue.len().max(1)).unwrap();
        let mut lru = Self {
            queue,
            capacity,
            next_priority: priority,
            _policy: std::marker::PhantomData,
        };
        lru.heapify();
        lru
    }
}
