use std::{fmt::Debug, hash::Hash};

use indexmap::IndexMap;

#[cfg(not(feature = "ahash"))]
type RandomState = std::hash::RandomState;
#[cfg(feature = "ahash")]
type RandomState = ahash::RandomState;

macro_rules! check_capacity_maybe_get {
    ($this:ident, $key:ident) => {
        if $this.queue.len() >= CAPACITY {
            if $this.queue.contains_key(&$key) {
                let index = $this.queue.get_index_of(&$key).unwrap();
                $this.queue[index].age = $this.age;
                $this.age -= 1;
                let index = $this.bubble_down(index);
                return &mut $this.queue[index].value;
            } else {
                $this.pop_internal();
            }
        }
    };
}

#[derive(Debug)]
struct Entry<Value> {
    age: u64,
    value: Value,
}

/// A fixed-capacity least-recently-used (LRU) cache.
///
/// The cache maintains at most `CAPACITY` entries. When inserting into a full cache,
/// the least recently used entry is automatically evicted. All operations that access
/// values (get, insert, get_or_insert_with) update the entry's position in the LRU order.
/// Operations that only read without accessing (peek, contains_key, oldest) do not
/// affect LRU ordering.
///
/// # Time Complexity
/// - Insert/Get/Remove: O(log n) average, O(n) worst case
/// - Peek/Contains: O(1) average, O(n) worst case  
/// - Pop (oldest): O(log n)
/// - Clear: O(1)
///
/// # Space Complexity
/// - O(CAPACITY) memory usage
/// - Pre-allocates space for CAPACITY entries
///
/// # Examples
///
/// ```
/// use evictor::Lru;
///
/// let mut cache = Lru::<3, i32, String>::new();
///
///
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
///
/// assert_eq!(cache.len(), 3);
///
///
/// cache.get(&1);
///
///
/// cache.insert(4, "four".to_string());
/// assert!(!cache.contains_key(&2));
///
///
/// cache.peek(&3);
/// let (oldest_key, _) = cache.oldest().unwrap();
/// assert_eq!(oldest_key, &3);
/// ```
///
/// # Capacity Requirements
///
/// The capacity must be greater than 0. This is enforced with a compile-time assertion.
/// ```compile_fail
/// # use evictor::Lru;
/// let _cache = Lru::<0, i32, String>::new();
/// ```
#[derive(Debug)]
pub struct Lru<const CAPACITY: usize, Key, Value> {
    queue: IndexMap<Key, Entry<Value>, RandomState>,
    age: u64,
}

impl<const CAPACITY: usize, Key, Value> Default for Lru<CAPACITY, Key, Value> {
    fn default() -> Self {
        Self {
            queue: IndexMap::with_capacity_and_hasher(CAPACITY, RandomState::default()),
            age: u64::MAX,
        }
    }
}

impl<const CAPACITY: usize, Key: Hash + Eq, Value> Lru<CAPACITY, Key, Value> {
    const CAPACITY_GT_ZERO: () = assert!(CAPACITY > 0, "Lru capacity must be greater than 0");

    /// Creates a new, empty LRU cache with the specified capacity.
    pub fn new() -> Self {
        let () = Self::CAPACITY_GT_ZERO;
        Self::default()
    }

    /// Removes all entries from the cache.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.age = u64::MAX;
    }

    /// Returns a reference to the value without updating its position in the cache.
    pub fn peek(&self, key: &Key) -> Option<&Value> {
        self.queue.get(key).map(|entry| &entry.value)
    }

    /// Returns a reference to the oldest (least recently used) entry.
    /// This does not update the entry's position in the cache.
    pub fn oldest(&self) -> Option<(&Key, &Value)> {
        self.queue.first().map(|(key, entry)| (key, &entry.value))
    }

    /// Returns true if the cache contains the given key.
    /// This does not update the entry's position in the cache.
    pub fn contains_key(&self, key: &Key) -> bool {
        self.queue.contains_key(key)
    }

    /// Gets the value for a key, or inserts it using the provided function.
    /// Returns an immutable reference to existing value (if found) or the newly inserted value.
    /// If the cache is full, the least recently used entry is removed.
    pub fn get_or_insert_with(
        &mut self,
        key: Key,
        or_insert: impl FnOnce(&Key) -> Value,
    ) -> &Value {
        self.get_or_insert_with_mut(key, or_insert)
    }

    /// Gets the value for a key, or inserts it using the provided function.
    /// Returns a mutable reference to the existing value (if found) or the newly inserted value.
    /// If the cache is full, the least recently used entry is removed.
    pub fn get_or_insert_with_mut(
        &mut self,
        key: Key,
        or_insert: impl FnOnce(&Key) -> Value,
    ) -> &mut Value {
        if self.age == 0 {
            self.age = u64::MAX;
            self.re_index(0);
        }

        check_capacity_maybe_get!(self, key);

        let index = match self.queue.entry(key) {
            indexmap::map::Entry::Occupied(o) => o.index(),
            indexmap::map::Entry::Vacant(v) => {
                let index = v.index();
                let e = Entry {
                    age: self.age,
                    value: or_insert(v.key()),
                };
                v.insert(e);
                index
            }
        };

        self.queue[index].age = self.age;
        self.age -= 1;
        let index = self.bubble_down(index);
        &mut self.queue[index].value
    }

    /// Inserts a key-value pair into the cache.
    /// Returns an immutable reference to the inserted value.
    /// If the cache is full, the least recently used entry is removed.
    pub fn insert(&mut self, key: Key, value: Value) -> &Value {
        self.insert_mut(key, value)
    }

    /// Inserts a key-value pair into the cache.
    /// Returns a mutable reference to the inserted value.
    /// If the cache is full, the least recently used entry is removed.
    pub fn insert_mut(&mut self, key: Key, value: Value) -> &mut Value {
        if self.age == 0 {
            self.age = u64::MAX;
            self.re_index(0);
        }

        check_capacity_maybe_get!(self, key);

        let index = self
            .queue
            .insert_full(
                key,
                Entry {
                    age: self.age,
                    value,
                },
            )
            .0;
        self.age -= 1;

        let index = self.bubble_down(index);

        &mut self.queue[index].value
    }

    /// Gets a value from the cache, marking it as recently used.
    /// Returns an immutable reference to the value if found.
    pub fn get(&mut self, key: &Key) -> Option<&Value> {
        self.get_mut(key).map(|v| &*v)
    }

    /// Gets a value from the cache, marking it as recently used.
    /// Returns a mutable reference to the value if found.
    pub fn get_mut(&mut self, key: &Key) -> Option<&mut Value> {
        if self.age == 0 {
            self.age = u64::MAX;
            self.re_index(0);
        }

        if let Some(mut index) = self.queue.get_index_of(key) {
            self.queue[index].age = self.age;
            self.age -= 1;

            index = self.bubble_down(index);

            Some(&mut self.queue[index].value)
        } else {
            None
        }
    }

    /// Removes and returns the oldest (least recently used) entry.
    pub fn pop(&mut self) -> Option<(Key, Value)> {
        self.pop_internal().map(|(key, entry)| (key, entry.value))
    }

    /// Removes a specific entry from the cache.
    /// Returns the value if the key was present.
    pub fn remove(&mut self, key: &Key) -> Option<Value> {
        if let Some(index) = self.queue.get_index_of(key) {
            let entry = self.queue.swap_remove_index(index);
            self.bubble_down(index);
            entry.map(|(_, entry)| entry.value)
        } else {
            None
        }
    }

    /// Returns true if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns the maximum number of entries the cache can hold.
    pub const fn capacity(&self) -> usize {
        CAPACITY
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
        self.bubble_down(0);
        result
    }

    fn bubble_down(&mut self, mut index: usize) -> usize {
        loop {
            let left_idx = index * 2 + 1;
            let right_idx = index * 2 + 2;

            if left_idx >= self.queue.len() {
                break;
            }

            if right_idx >= self.queue.len() {
                if self.queue[left_idx].age > self.queue[index].age {
                    self.queue.swap_indices(index, left_idx);
                    index = left_idx;
                }
                break;
            }

            let target = if self.queue[left_idx].age > self.queue[right_idx].age {
                left_idx
            } else {
                right_idx
            };

            if self.queue[target].age < self.queue[index].age {
                break;
            }

            self.queue.swap_indices(index, target);
            index = target;
        }

        index
    }

    fn re_index(&mut self, index: usize) {
        debug_assert!(self.age > 0);
        if index >= self.queue.len() {
            return;
        }

        self.queue[index].age = self.age;
        self.age -= 1;

        let left_idx = index * 2 + 1;
        let right_idx = index * 2 + 2;

        self.re_index(left_idx);
        self.re_index(right_idx);
    }

    fn heapify(&mut self) {
        if self.queue.is_empty() {
            return;
        }

        for i in (0..self.queue.len()).rev() {
            self.bubble_down(i);
        }
    }
}

impl<const CAPACITY: usize, Key: Hash + Eq, Value> std::iter::FromIterator<(Key, Value)>
    for Lru<CAPACITY, Key, Value>
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Key, Value)>,
    {
        let mut queue = IndexMap::with_capacity_and_hasher(CAPACITY, RandomState::default());
        let mut age = u64::MAX;

        for (key, value) in iter {
            queue.insert(key, Entry { age, value });
            age -= 1;
        }

        if queue.len() > CAPACITY {
            queue.reverse();
            queue.truncate(CAPACITY);
        }

        let mut lru = Self { queue, age };
        lru.heapify();
        lru
    }
}

#[cfg(test)]
mod tests {

    use indexmap::IndexMap;

    use crate::{Lru, RandomState};

    #[test]
    fn trivial() {
        let mut lru = Lru::<5, _, _>::default();

        for key in 0..5 {
            lru.insert(key, key);
        }

        let mut vals = vec![];
        while let Some((_, v)) = lru.pop() {
            vals.push(v);
        }
    }

    #[test]
    fn test_capacity() {
        let mut lru = Lru::<5, _, _>::default();

        for key in 0..10 {
            lru.insert(key, key);
            if lru.len() > 5 {
                assert!(lru.queue.contains_key(&(key - 5)));
            }
        }

        assert_eq!(lru.len(), 5);
        for key in 5..10 {
            assert!(lru.queue.contains_key(&key));
        }
    }

    #[test]
    fn test_clear() {
        let mut lru = Lru::<3, i32, String>::default();
        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());
        lru.insert(3, "three".to_string());

        assert_eq!(lru.len(), 3);
        assert!(!lru.is_empty());

        lru.clear();

        assert_eq!(lru.len(), 0);
        assert!(lru.is_empty());
        assert_eq!(lru.peek(&1), None);
        assert_eq!(lru.peek(&2), None);
        assert_eq!(lru.peek(&3), None);
    }

    #[test]
    fn test_peek() {
        let mut lru = Lru::<3, i32, String>::default();
        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());

        assert_eq!(lru.peek(&1), Some(&"one".to_string()));
        assert_eq!(lru.peek(&2), Some(&"two".to_string()));
        assert_eq!(lru.peek(&3), None);

        lru.insert(3, "three".to_string());
        lru.insert(4, "four".to_string());

        assert_eq!(lru.peek(&1), None);
        assert_eq!(lru.peek(&2), Some(&"two".to_string()));
        assert_eq!(lru.peek(&3), Some(&"three".to_string()));
        assert_eq!(lru.peek(&4), Some(&"four".to_string()));
    }

    #[test]
    fn test_oldest() {
        let mut lru = Lru::<3, i32, String>::default();

        assert_eq!(lru.oldest(), None);

        lru.insert(1, "one".to_string());
        assert_eq!(lru.oldest(), Some((&1, &"one".to_string())));

        lru.insert(2, "two".to_string());
        assert_eq!(lru.oldest(), Some((&1, &"one".to_string())));

        lru.get(&1);
        assert_eq!(lru.oldest(), Some((&2, &"two".to_string())));
    }

    #[test]
    fn test_contains_key() {
        let mut lru = Lru::<3, i32, String>::default();

        assert!(!lru.contains_key(&1));

        lru.insert(1, "one".to_string());
        assert!(lru.contains_key(&1));
        assert!(!lru.contains_key(&2));

        lru.remove(&1);
        assert!(!lru.contains_key(&1));
    }

    #[test]
    fn test_get_or_insert_with() {
        let mut lru = Lru::<3, i32, String>::default();
        let mut call_count = 0;

        let value = lru.get_or_insert_with(1, |_| {
            call_count += 1;
            "one".to_string()
        });
        assert_eq!(value, &"one".to_string());
        assert_eq!(call_count, 1);

        let value = lru.get_or_insert_with(1, |_| {
            call_count += 1;
            "different".to_string()
        });
        assert_eq!(value, &"one".to_string());
        assert_eq!(call_count, 1);
    }

    #[test]
    fn test_get_or_insert_with_mut() {
        let mut lru = Lru::<3, i32, String>::default();

        let value = lru.get_or_insert_with_mut(1, |_| "one".to_string());
        *value = "modified".to_string();

        assert_eq!(lru.peek(&1), Some(&"modified".to_string()));

        let value = lru.get_or_insert_with_mut(1, |_| "different".to_string());
        assert_eq!(value, &"modified".to_string());
    }

    #[test]
    fn test_insert_and_insert_mut() {
        let mut lru = Lru::<3, i32, String>::default();

        let value = lru.insert(1, "one".to_string());
        assert_eq!(value, &"one".to_string());

        let value = lru.insert_mut(2, "two".to_string());
        *value = "modified_two".to_string();

        assert_eq!(lru.peek(&1), Some(&"one".to_string()));
        assert_eq!(lru.peek(&2), Some(&"modified_two".to_string()));

        lru.insert(1, "new_one".to_string());
        assert_eq!(lru.peek(&1), Some(&"new_one".to_string()));
    }

    #[test]
    fn test_get_and_get_mut() {
        let mut lru = Lru::<3, i32, String>::default();
        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());

        assert_eq!(lru.get(&1), Some(&"one".to_string()));
        assert_eq!(lru.get(&3), None);

        if let Some(value) = lru.get_mut(&2) {
            *value = "modified_two".to_string();
        }

        assert_eq!(lru.peek(&2), Some(&"modified_two".to_string()));
        assert_eq!(lru.get_mut(&3), None);
    }

    #[test]
    fn test_pop() {
        let mut lru = Lru::<3, i32, String>::default();

        assert_eq!(lru.pop(), None);

        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());
        lru.insert(3, "three".to_string());

        let (key, value) = lru.pop().unwrap();
        assert_eq!(key, 1);
        assert_eq!(value, "one".to_string());
        assert_eq!(lru.len(), 2);

        lru.get(&2);
        let (key, _) = lru.pop().unwrap();
        assert_eq!(key, 3);
    }

    #[test]
    fn test_remove() {
        let mut lru = Lru::<3, i32, String>::default();
        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());

        assert_eq!(lru.remove(&1), Some("one".to_string()));
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.remove(&1), None);
        assert_eq!(lru.remove(&3), None);

        assert_eq!(lru.peek(&1), None);
        assert_eq!(lru.peek(&2), Some(&"two".to_string()));
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut lru = Lru::<3, i32, String>::default();

        assert!(lru.is_empty());
        assert_eq!(lru.len(), 0);

        lru.insert(1, "one".to_string());
        assert!(!lru.is_empty());
        assert_eq!(lru.len(), 1);

        lru.insert(2, "two".to_string());
        assert_eq!(lru.len(), 2);

        lru.clear();
        assert!(lru.is_empty());
        assert_eq!(lru.len(), 0);
    }

    #[test]
    fn test_capacity_method() {
        let lru = Lru::<5, i32, String>::default();
        assert_eq!(lru.capacity(), 5);

        let lru = Lru::<100, i32, String>::default();
        assert_eq!(lru.capacity(), 100);
    }

    #[test]
    fn test_retain() {
        let mut lru = Lru::<5, i32, i32>::default();
        for i in 1..=5 {
            lru.insert(i, i * 10);
        }

        lru.retain(|&key, _| key % 2 == 0);

        assert_eq!(lru.len(), 2);
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&4));
        assert!(!lru.contains_key(&1));
        assert!(!lru.contains_key(&3));
        assert!(!lru.contains_key(&5));

        lru.retain(|_, value| {
            *value *= 2;
            true
        });

        assert_eq!(lru.peek(&2), Some(&40));
        assert_eq!(lru.peek(&4), Some(&80));
    }

    #[test]
    fn test_extend() {
        let mut lru = Lru::<5, i32, String>::default();
        lru.insert(1, "one".to_string());

        let items = vec![
            (2, "two".to_string()),
            (3, "three".to_string()),
            (4, "four".to_string()),
        ];

        lru.extend(items);

        assert_eq!(lru.len(), 4);
        assert_eq!(lru.peek(&1), Some(&"one".to_string()));
        assert_eq!(lru.peek(&2), Some(&"two".to_string()));
        assert_eq!(lru.peek(&3), Some(&"three".to_string()));
        assert_eq!(lru.peek(&4), Some(&"four".to_string()));

        lru.extend(vec![(5, "five".to_string()), (6, "six".to_string())]);
        assert_eq!(lru.len(), 5);
        assert_eq!(lru.peek(&6), Some(&"six".to_string()));
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut lru = Lru::<10, i32, String>::default();
        for i in 1..=5 {
            lru.insert(i, format!("value_{}", i));
        }

        lru.remove(&1);
        lru.remove(&2);

        assert_eq!(lru.len(), 3);
        lru.shrink_to_fit();

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.peek(&3), Some(&"value_3".to_string()));
        assert_eq!(lru.peek(&4), Some(&"value_4".to_string()));
        assert_eq!(lru.peek(&5), Some(&"value_5".to_string()));
    }

    #[test]
    fn test_lru_order_complex() {
        let mut lru = Lru::<3, i32, String>::default();

        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());
        lru.insert(3, "three".to_string());

        lru.get(&1);
        lru.get(&3);

        lru.insert(4, "four".to_string());

        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&3));
        assert!(lru.contains_key(&4));

        let (key1, _) = lru.pop().unwrap();
        assert_eq!(key1, 1);

        let (key2, _) = lru.pop().unwrap();
        assert_eq!(key2, 3);

        let (key3, _) = lru.pop().unwrap();
        assert_eq!(key3, 4);
    }

    #[test]
    fn test_get_or_insert_capacity_behavior() {
        let mut lru = Lru::<2, i32, String>::default();

        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());

        lru.get_or_insert_with(1, |_| "new_one".to_string());
        assert_eq!(lru.len(), 2);
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));

        lru.get_or_insert_with(3, |_| "three".to_string());
        assert_eq!(lru.len(), 2);
        assert!(lru.contains_key(&1));
        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&3));
    }

    #[test]
    fn test_edge_cases() {
        let mut lru = Lru::<1, i32, String>::default();
        lru.insert(1, "one".to_string());
        assert_eq!(lru.len(), 1);

        lru.insert(2, "two".to_string());
        assert_eq!(lru.len(), 1);
        assert!(!lru.contains_key(&1));
        assert!(lru.contains_key(&2));

        let mut lru = Lru::<3, i32, String>::default();
        assert_eq!(lru.get(&1), None);
        assert_eq!(lru.remove(&1), None);
        assert_eq!(lru.pop(), None);
        assert_eq!(lru.oldest(), None);
    }

    #[test]
    fn test_from_iter_basic() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let lru: Lru<5, i32, String> = items.into_iter().collect();

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.peek(&1), Some(&"one".to_string()));
        assert_eq!(lru.peek(&2), Some(&"two".to_string()));
        assert_eq!(lru.peek(&3), Some(&"three".to_string()));

        assert_eq!(lru.oldest(), Some((&1, &"one".to_string())));
    }

    #[test]
    fn test_from_iter_capacity_enforcement() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
            (4, "four".to_string()),
            (5, "five".to_string()),
        ];

        let lru: Lru<3, i32, String> = items.into_iter().collect();

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.peek(&1), None);
        assert_eq!(lru.peek(&2), None);
        assert_eq!(lru.peek(&3), Some(&"three".to_string()));
        assert_eq!(lru.peek(&4), Some(&"four".to_string()));
        assert_eq!(lru.peek(&5), Some(&"five".to_string()));

        assert_eq!(lru.oldest(), Some((&3, &"three".to_string())));
    }

    #[test]
    fn test_from_iter_empty() {
        let items: Vec<(i32, String)> = vec![];
        let lru: Lru<5, i32, String> = items.into_iter().collect();

        assert!(lru.is_empty());
        assert_eq!(lru.len(), 0);
        assert_eq!(lru.oldest(), None);
    }

    #[test]
    fn test_from_iter_single_item() {
        let items = vec![(42, "answer".to_string())];
        let lru: Lru<10, i32, String> = items.into_iter().collect();

        assert_eq!(lru.len(), 1);
        assert_eq!(lru.peek(&42), Some(&"answer".to_string()));
        assert_eq!(lru.oldest(), Some((&42, &"answer".to_string())));
    }

    #[test]
    fn test_from_iter_lru_order() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let mut lru: Lru<3, i32, String> = items.into_iter().collect();

        let (key1, value1) = lru.pop().unwrap();
        assert_eq!(key1, 1);
        assert_eq!(value1, "one".to_string());

        let (key2, value2) = lru.pop().unwrap();
        assert_eq!(key2, 2);
        assert_eq!(value2, "two".to_string());

        let (key3, value3) = lru.pop().unwrap();
        assert_eq!(key3, 3);
        assert_eq!(value3, "three".to_string());

        assert!(lru.is_empty());
    }

    #[test]
    fn test_duplicate_key_insertions() {
        let mut lru = Lru::<3, i32, String>::default();

        lru.insert(1, "first".to_string());
        lru.insert(2, "second".to_string());
        lru.insert(1, "updated".to_string());

        assert_eq!(lru.len(), 2);
        assert_eq!(lru.peek(&1), Some(&"updated".to_string()));
        assert_eq!(lru.peek(&2), Some(&"second".to_string()));

        lru.insert(3, "third".to_string());
        lru.insert(4, "fourth".to_string());

        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&3));
        assert!(lru.contains_key(&4));
    }

    #[test]
    fn test_age_counter_stress() {
        let mut lru = Lru::<100, i32, i32> {
            queue: IndexMap::with_capacity_and_hasher(100, RandomState::default()),
            age: 50,
        };

        for i in 0..200 {
            lru.insert(i, i * 2);
            if i % 20 == 0 {
                lru.get(&(i / 2));
            }
        }

        assert_eq!(lru.len(), 100);

        let mut prev_age = u64::MAX;
        while let Some((_, entry)) = lru.pop_internal() {
            assert!(entry.age <= prev_age, "Age order violated");
            prev_age = entry.age;
        }
    }

    #[test]
    fn test_remove_from_different_positions() {
        let mut lru = Lru::<5, i32, String>::default();

        for i in 1..=5 {
            lru.insert(i, format!("value_{}", i));
        }

        lru.get(&3);
        lru.get(&1);
        lru.get(&4);

        assert_eq!(lru.remove(&3), Some("value_3".to_string()));
        assert_eq!(lru.len(), 4);

        let oldest = *lru.oldest().unwrap().0;
        assert_eq!(lru.remove(&oldest), Some(format!("value_{}", oldest)));
        assert_eq!(lru.len(), 3);

        let mut ages = vec![];
        while let Some((_, entry)) = lru.pop_internal() {
            ages.push(entry.age);
        }

        for window in ages.windows(2) {
            assert!(window[0] >= window[1], "Heap property violated");
        }
    }

    #[test]
    fn test_extend_with_overlapping_keys() {
        let mut lru = Lru::<4, i32, String>::default();

        lru.insert(1, "original_1".to_string());
        lru.insert(2, "original_2".to_string());

        let items = vec![
            (2, "updated_2".to_string()),
            (3, "new_3".to_string()),
            (4, "new_4".to_string()),
            (5, "new_5".to_string()),
        ];

        lru.extend(items);

        assert_eq!(lru.len(), 4);
        assert_eq!(lru.peek(&2), Some(&"updated_2".to_string()));
        assert_eq!(lru.peek(&3), Some(&"new_3".to_string()));
        assert_eq!(lru.peek(&4), Some(&"new_4".to_string()));
        assert_eq!(lru.peek(&5), Some(&"new_5".to_string()));

        assert!(!lru.contains_key(&1));
    }

    #[test]
    fn test_alternating_access_patterns() {
        let mut lru = Lru::<4, i32, String>::default();

        for i in 1..=4 {
            lru.insert(i, format!("value_{}", i));
        }

        for _ in 0..10 {
            lru.get(&1);
            lru.get(&3);
            lru.get(&2);
            lru.get(&4);
        }

        for i in 1..=4 {
            assert!(lru.contains_key(&i));
        }

        lru.insert(5, "value_5".to_string());
        assert_eq!(lru.len(), 4);

        let mut count = 0;
        for i in 1..=5 {
            if lru.contains_key(&i) {
                count += 1;
            }
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn test_large_capacity_operations() {
        let mut lru = Lru::<1000, usize, usize>::default();

        for i in 0..1000 {
            lru.insert(i, i * i);
        }

        for i in 0..500 {
            lru.get(&(i * 2 % 1000));
        }

        for i in 0..100 {
            lru.remove(&(i * 3 % 1000));
        }

        for i in 1000..1100 {
            lru.insert(i, i * i);
        }

        assert!(lru.len() <= 1000);

        let mut prev_age = u64::MAX;
        while let Some((_, entry)) = lru.pop_internal() {
            assert!(entry.age <= prev_age, "Heap order violated in large cache");
            prev_age = entry.age;
        }
    }
}
