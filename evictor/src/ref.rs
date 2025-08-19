use std::ops::{
    Deref,
    DerefMut,
};

use crate::{
    Cache,
    EntryValue,
    Policy,
};

/// A smart reference to a cached value that tracks modifications.
///
/// The `Entry` provides transparent access to the underlying value through
/// `Deref` and `DerefMut` traits.
///
/// # Behavior
///
/// ## Automatic Eviction Order Updates
/// When an `Entry` is dropped:
/// - If the value was **modified** during the borrow (via `DerefMut`, `AsMut`,
///   or `value_mut()`), the cache's eviction order is updated
/// - If the value was **never modified**, the eviction order remains unchanged
///
/// ## Modification Tracking
/// The `Entry` tracks modifications through several mechanisms:
/// - **`DerefMut`**: Any mutable dereference (`*entry = new_value`) marks as
///   dirty
/// - **`AsMut`**: Calling `entry.as_mut()` marks as dirty
/// - **`value_mut()`**: Calling `entry.value_mut()` marks as dirty
/// - **Read-only access**: `Deref`, `AsRef`, `value()`, and `key()` do not mark
///   as dirty
///
/// # Performance
///
/// - **Lookup**: O(1) average (inherits from hash map performance)
/// - **No modification**: Zero additional overhead beyond the initial lookup
/// - **With modification**: O(1) cache reordering when dropped
///
/// # Examples
///
/// ## Read-only Access (No Eviction Order Change)
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
/// // Read-only access through Entry
/// if let Some(entry) = cache.peek_mut(&"A") {
///     let _len = entry.len(); // Read-only
///     let _first = entry.first(); // Read-only
///     let _key = entry.key(); // Read-only
///     let _value_ref = entry.value(); // Read-only
/// } // Entry dropped here - no modifications, so no cache update
///
/// let new_order: Vec<_> = cache.iter().map(|(k, _)| *k).collect();
/// assert_eq!(original_order, new_order); // Order unchanged
/// ```
///
/// ## Modification Triggers Cache Update
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
/// // Before: "A" would be evicted first
/// assert_eq!(cache.tail().unwrap().0, &"A");
///
/// // Modify "A" through Entry
/// if let Some(mut entry) = cache.peek_mut(&"A") {
///     entry.push(4); // Modification via DerefMut
/// } // Entry dropped here - modification detected, cache updated
///
/// // After: "A" is now most recently used, "B" would be evicted first
/// assert_eq!(cache.tail().unwrap().0, &"B");
/// ```
///
/// ## Different Modification Methods
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::Lru;
///
/// let mut cache = Lru::new(NonZeroUsize::new(2).unwrap());
/// cache.insert("key", String::from("hello"));
///
/// // Method 1: Direct assignment via DerefMut
/// if let Some(mut entry) = cache.peek_mut(&"key") {
///     *entry = String::from("world"); // Triggers modification tracking
/// }
///
/// // Method 2: Mutable method call via DerefMut
/// if let Some(mut entry) = cache.peek_mut(&"key") {
///     entry.push_str(" rust"); // Triggers modification tracking
/// }
///
/// // Method 3: Explicit mutable reference
/// if let Some(mut entry) = cache.peek_mut(&"key") {
///     let value = entry.value_mut(); // Triggers modification tracking
///     value.push('!');
/// }
///
/// // Method 4: AsMut trait
/// if let Some(mut entry) = cache.peek_mut(&"key") {
///     entry.as_mut().push('?'); // Triggers modification tracking
/// }
/// ```
pub struct Entry<'c, K, V, P: Policy<V>> {
    index: usize,
    dirty: bool,
    cache: &'c mut Cache<K, V, P>,
}

impl<K, V, P: Policy<V>> Drop for Entry<'_, K, V, P> {
    fn drop(&mut self) {
        if self.dirty {
            P::touch_entry(
                self.index,
                false,
                &mut self.cache.metadata,
                &mut self.cache.queue,
            );

            #[cfg(feature = "statistics")]
            {
                self.cache.statistics.hits += 1;
            }
        }
    }
}

impl<K, V, P: Policy<V>> AsRef<V> for Entry<'_, K, V, P> {
    fn as_ref(&self) -> &V {
        self.cache.queue[self.index].value()
    }
}

impl<K, V, P: Policy<V>> AsMut<V> for Entry<'_, K, V, P> {
    fn as_mut(&mut self) -> &mut V {
        self.dirty = true;
        self.cache.queue[self.index].value_mut()
    }
}

impl<K, V, P: Policy<V>> Deref for Entry<'_, K, V, P> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        self.cache.queue[self.index].value()
    }
}

impl<K, V, P: Policy<V>> DerefMut for Entry<'_, K, V, P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dirty = true;
        self.cache.queue[self.index].value_mut()
    }
}

impl<'q, K, V, P: Policy<V>> Entry<'q, K, V, P> {
    pub(crate) fn new(index: usize, cache: &'q mut Cache<K, V, P>) -> Self {
        Self {
            index,
            dirty: false,
            cache,
        }
    }
}

impl<K, V, P: Policy<V>> Entry<'_, K, V, P> {
    /// Returns a reference to the key for this cache entry.
    ///
    /// This method provides read-only access to the key associated with the
    /// cached value. The key reference is valid for the lifetime of the `Entry`
    /// and accessing it does not affect the cache's eviction order or mark the
    /// entry as modified.
    ///
    /// # Returns
    ///
    /// A reference to the key of type `&K`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("hello", vec![1, 2, 3]);
    ///
    /// if let Some(entry) = cache.peek_mut(&"hello") {
    ///     assert_eq!(entry.key(), &"hello");
    ///     // Key access doesn't mark entry as dirty
    /// }
    /// ```
    pub fn key(&self) -> &K {
        self.cache
            .queue
            .get_index(self.index)
            .expect("Entry index out of bounds")
            .0
    }

    /// Returns an immutable reference to the cached value.
    ///
    /// This method provides read-only access to the cached value without
    /// affecting the cache's eviction order or marking the entry as modified.
    /// It's equivalent to using the `Deref` trait (e.g., `&*entry`) or the
    /// `AsRef` trait.
    ///
    /// For mutable access that tracks modifications, use [`value_mut()`]
    /// instead.
    ///
    /// # Returns
    ///
    /// An immutable reference to the cached value of type `&V`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
    /// cache.insert("key", vec![1, 2, 3]);
    ///
    /// if let Some(entry) = cache.peek_mut(&"key") {
    ///     let value_ref = entry.value();
    ///     assert_eq!(value_ref, &vec![1, 2, 3]);
    ///
    ///     // These are equivalent ways to access the value:
    ///     assert_eq!(entry.value(), &*entry);
    ///     assert_eq!(entry.value(), entry.as_ref());
    /// }
    /// ```
    ///
    /// [`value_mut()`]: Entry::value_mut
    pub fn value(&self) -> &V {
        self.cache.queue[self.index].value()
    }

    /// Returns a mutable reference to the cached value and marks the entry as
    /// modified.
    ///
    /// This method provides mutable access to the cached value and
    /// automatically marks the entry as "dirty", indicating that it has
    /// been modified. When the `Entry` is dropped, the cache's eviction
    /// order will be updated to reflect this modification according to the
    /// cache policy.
    ///
    /// **Important**: Unlike `DerefMut` which only marks as dirty when actually
    /// dereferenced mutably, this method **always** marks the entry as dirty,
    /// even if you don't actually modify the returned reference.
    ///
    /// For read-only access that doesn't affect eviction order, use [`value()`]
    /// instead.
    ///
    /// # Returns
    ///
    /// A mutable reference to the cached value of type `&mut V`.
    ///
    /// # Behavior
    ///
    /// - Immediately marks the entry as modified (dirty flag set to `true`)
    /// - When the `Entry` is dropped, the cache's eviction order will be
    ///   updated
    /// - The specific eviction order update depends on the cache policy (Lru,
    ///   Mru, Lfu)
    ///
    /// # Examples
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
    /// // Before modification: "A" would be evicted first (least recently used)
    /// assert_eq!(cache.tail().unwrap().0, &"A");
    ///
    /// if let Some(mut entry) = cache.peek_mut(&"A") {
    ///     let value = entry.value_mut(); // Marks as dirty immediately
    ///     value.push(4); // Modify the value
    /// } // Entry dropped here - cache eviction order updated
    ///
    /// // After modification: "A" is now most recently used
    /// assert_eq!(cache.tail().unwrap().0, &"B");
    /// ```
    ///
    /// ## Comparison with Other Access Methods
    ///
    /// ```rust
    /// use std::num::NonZeroUsize;
    ///
    /// use evictor::Lru;
    ///
    /// let mut cache = Lru::new(NonZeroUsize::new(2).unwrap());
    /// cache.insert("key", String::from("hello"));
    ///
    /// if let Some(mut entry) = cache.peek_mut(&"key") {
    ///     // Method 1: value_mut() - always marks as dirty
    ///     let value = entry.value_mut();
    ///     value.push_str(" world");
    ///
    ///     // Method 2: DerefMut - only marks dirty when actually used mutably
    ///     entry.push_str("!"); // Equivalent to (*entry).push_str("!")
    ///
    ///     // Method 3: AsMut trait
    ///     entry.as_mut().push_str("?");
    /// }
    /// ```
    ///
    /// [`value()`]: Entry::value
    pub fn value_mut(&mut self) -> &mut V {
        self.dirty = true;
        self.cache.queue[self.index].value_mut()
    }
}
