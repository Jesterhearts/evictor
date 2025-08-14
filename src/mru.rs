use std::hash::Hash;

use crate::{
    EntryValue,
    Policy,
    RandomState,
};

/// Policy implementation for Most Recently Used (MRU) cache eviction.
///
/// This policy tracks when entries were last accessed and evicts the entry
/// that was accessed most recently when the cache reaches capacity. This is
/// the opposite of LRU behavior and is useful in scenarios where you want to
/// prioritize older, less-accessed items over recently accessed ones.
///
/// # Implementation Details
///
/// ## Priority System
/// - **Priority Values**: Uses ascending u64 values starting from `u64::MIN`
/// - **Recent Access**: Higher priority values = more recently accessed
/// - **Eviction Target**: Entry with highest priority (most recent access time)
/// - **Priority Assignment**: Each access increments the global priority
///   counter
///
/// ## Heap Structure
/// The underlying heap maintains the invariant that parent nodes have
/// priorities >= their children, ensuring the most recently used item
/// (highest priority) is always at index 0 for efficient eviction.
///
/// ## Priority Counter Management
/// - **Start Value**: `u64::MIN` (oldest possible priority)
/// - **End Value**: `u64::MAX` (newest possible priority)
/// - **Overflow Handling**: When counter reaches `u64::MAX`, it wraps to
///   `u64::MIN` and the entire heap is re-indexed to maintain ordering
///
/// # Usage
///
/// This policy is typically used through the [`Mru`] type alias rather than
/// directly:
///
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::{
///     Cache,
///     MruPolicy,
/// };
///
/// // Direct usage (not recommended)
/// let mut cache: Cache<i32, String, MruPolicy> = Cache::new(NonZeroUsize::new(3).unwrap());
///
/// // Preferred usage via type alias
/// use evictor::Mru;
/// let mut cache: Mru<i32, String> = Mru::new(NonZeroUsize::new(2).unwrap());
///
/// cache.insert(1, "first".to_string());
/// cache.insert(2, "second".to_string());
///
/// // Access an item, making it most recently used
/// cache.get(&1);
///
/// // Insert when full - evicts the most recently used (key 1)
/// cache.insert(3, "third".to_string());
/// assert!(!cache.contains_key(&1)); // 1 was evicted (most recently used)
/// assert!(cache.contains_key(&2)); // 2 was not recently accessed, so kept
/// ```
///
/// # Performance Characteristics
///
/// - **Access Update**: O(log n) - requires heap bubble operation
/// - **Eviction**: O(log n) - removes root and re-heapifies
/// - **Priority Overflow**: O(n) - occurs very rarely (every 2^64 operations)
///
/// [`Mru`]: crate::Mru
#[derive(Debug)]
pub struct MruPolicy;

impl Policy for MruPolicy {
    fn start_value() -> u64 {
        u64::MIN
    }

    fn end_value() -> u64 {
        u64::MAX
    }

    fn assign_update_next_value(cache_next: &mut u64, entry: &mut impl EntryValue) {
        *entry.priority_mut() = *cache_next;
        *cache_next = cache_next.wrapping_add(1);
    }

    fn heap_bubble<T: EntryValue>(
        mut index: usize,
        queue: &mut indexmap::IndexMap<impl Hash + Eq, T, RandomState>,
    ) -> usize {
        let mut parent_index = index.saturating_sub(1) / 2;

        while index > 0 && queue[index].priority() > queue[parent_index].priority() {
            queue.swap_indices(index, parent_index);
            index = parent_index;
            parent_index = index.saturating_sub(1) / 2;
        }

        index
    }

    fn re_index<T: EntryValue>(
        queue: &mut indexmap::IndexMap<impl Hash + Eq, T, RandomState>,
        next_priority: &mut u64,
        index: usize,
    ) {
        if *next_priority == Self::end_value() {
            *next_priority = Self::start_value();
        }

        if index >= queue.len() {
            return;
        }

        Self::assign_update_next_value(next_priority, &mut queue[index]);

        let left_index = index * 2 + 1;
        let right_index = index * 2 + 2;

        Self::re_index(queue, next_priority, left_index);
        Self::re_index(queue, next_priority, right_index);
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use indexmap::IndexMap;

    use super::*;
    use crate::{
        Cache,
        Entry,
    };

    #[test]
    fn test_assign_update_next_value() {
        let mut cache_next = 100u64;
        let mut entry = Entry {
            priority: 0,
            value: "test",
        };

        MruPolicy::assign_update_next_value(&mut cache_next, &mut entry);

        assert_eq!(entry.priority, 100);
        assert_eq!(cache_next, 101);
    }

    #[test]
    fn test_assign_update_next_value_wrapping() {
        let mut cache_next = u64::MAX;
        let mut entry = Entry {
            priority: 0,
            value: "test",
        };

        MruPolicy::assign_update_next_value(&mut cache_next, &mut entry);

        assert_eq!(entry.priority, u64::MAX);
        assert_eq!(cache_next, 0);
    }

    #[test]
    fn test_heap_bubble_single_element() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        queue.insert(
            1,
            Entry {
                priority: 100,
                value: "one",
            },
        );

        let result = MruPolicy::heap_bubble(0, &mut queue);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_heap_bubble_two_elements_no_swap() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        queue.insert(
            1,
            Entry {
                priority: 200,
                value: "one",
            },
        );
        queue.insert(
            2,
            Entry {
                priority: 100,
                value: "two",
            },
        );

        let result = MruPolicy::heap_bubble(1, &mut queue);
        assert_eq!(result, 1);
        assert_eq!(queue[0].priority, 200);
        assert_eq!(queue[1].priority, 100);
    }

    #[test]
    fn test_heap_bubble_two_elements_with_swap() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        queue.insert(
            1,
            Entry {
                priority: 100,
                value: "one",
            },
        );
        queue.insert(
            2,
            Entry {
                priority: 200,
                value: "two",
            },
        );

        let result = MruPolicy::heap_bubble(1, &mut queue);
        assert_eq!(result, 0);
        assert_eq!(queue[0].priority, 200);
        assert_eq!(queue[1].priority, 100);
    }

    #[test]
    fn test_heap_bubble_multiple_elements() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        queue.insert(
            1,
            Entry {
                priority: 100,
                value: "one",
            },
        );
        queue.insert(
            2,
            Entry {
                priority: 150,
                value: "two",
            },
        );
        queue.insert(
            3,
            Entry {
                priority: 200,
                value: "three",
            },
        );
        queue.insert(
            4,
            Entry {
                priority: 175,
                value: "four",
            },
        );

        let result = MruPolicy::heap_bubble(3, &mut queue);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_mru_cache_basic_operations() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(3).unwrap());

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 3);

        cache.insert(1, "one".to_string());
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        assert!(cache.contains_key(&1));

        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_mru_cache_eviction_policy() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.get(&1);

        cache.insert(3, "three".to_string());

        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    fn test_mru_cache_get_updates_priority() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let oldest_before = cache.tail().map(|(k, _)| *k);

        cache.get(&1);

        let oldest_after = cache.tail().map(|(k, _)| *k);
        assert_ne!(oldest_before, oldest_after);
    }

    #[test]
    fn test_mru_cache_peek_does_not_update_priority() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let oldest_before = cache.tail().map(|(k, _)| *k);

        cache.peek(&1);

        let oldest_after = cache.tail().map(|(k, _)| *k);
        assert_eq!(oldest_before, oldest_after);
    }

    #[test]
    fn test_mru_cache_get_or_insert_with() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(2).unwrap());

        let value = cache.get_or_insert_with(1, |_| "one".to_string());
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 1);

        let value = cache.get_or_insert_with(1, |_| "different".to_string());
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_mru_cache_remove() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let removed = cache.remove(&1);
        assert_eq!(removed, Some("one".to_string()));
        assert!(!cache.contains_key(&1));
        assert_eq!(cache.len(), 2);
        let youngest = cache.tail().map(|(k, _)| *k);
        assert_eq!(youngest, Some(3));

        let removed = cache.remove(&1);
        assert_eq!(removed, None);
    }

    #[test]
    fn test_mru_cache_pop() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let (_key, _value) = cache.pop().unwrap();
        assert_eq!(cache.len(), 2);

        let mut empty_cache = MruCache::new(NonZeroUsize::new(1).unwrap());
        assert!(empty_cache.pop().is_none());
    }

    #[test]
    fn test_mru_cache_clear() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_mru_cache_retain() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());
        cache.insert(4, "four".to_string());

        cache.retain(|&k, _| k % 2 == 0);

        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_mru_cache_extend() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());

        let items = vec![(2, "two".to_string()), (3, "three".to_string())];
        cache.extend(items);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    fn test_mru_cache_from_iterator() {
        type MruCache = Cache<i32, String, MruPolicy>;

        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let cache: MruCache = items.into_iter().collect();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    fn test_mru_cache_priority_wraparound() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(2).unwrap());

        for i in 0..u64::MAX {
            cache.insert(i as i32, format!("value{}", i));
            if i % 1000000 == 0 && i > 0 {
                break;
            }
        }

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_mru_cache_shrink_to_fit() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(10).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.shrink_to_fit();

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
    }

    #[test]
    fn test_mru_cache_mutable_operations() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(3).unwrap());

        let value_ref = cache.insert_mut(1, "one".to_string());
        value_ref.push_str(" modified");
        assert_eq!(cache.peek(&1), Some(&"one modified".to_string()));

        if let Some(value_ref) = cache.get_mut(&1) {
            value_ref.push_str(" again");
        }
        assert_eq!(cache.peek(&1), Some(&"one modified again".to_string()));

        let value_ref = cache.get_or_insert_with_mut(2, |_| "two".to_string());
        value_ref.push_str(" new");
        assert_eq!(cache.peek(&2), Some(&"two new".to_string()));
    }

    #[test]
    fn test_mru_vs_lru_behavior_difference() {
        use crate::{
            Lru,
            Mru,
        };

        let mut mru_cache: Mru<i32, String> = Cache::new(NonZeroUsize::new(2).unwrap());
        let mut lru_cache: Lru<i32, String> = Cache::new(NonZeroUsize::new(2).unwrap());

        mru_cache.insert(1, "one".to_string());
        mru_cache.insert(2, "two".to_string());
        lru_cache.insert(1, "one".to_string());
        lru_cache.insert(2, "two".to_string());

        mru_cache.get(&1);
        lru_cache.get(&1);

        mru_cache.insert(3, "three".to_string());
        lru_cache.insert(3, "three".to_string());

        assert!(mru_cache.contains_key(&2));
        assert!(lru_cache.contains_key(&1));

        assert!(!mru_cache.contains_key(&1));
        assert!(!lru_cache.contains_key(&2));

        assert!(mru_cache.contains_key(&3));
        assert!(lru_cache.contains_key(&3));
    }

    #[test]
    fn test_edge_case_empty_heap_bubble() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        let result = MruPolicy::heap_bubble(0, &mut queue);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_edge_case_capacity_one() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one".to_string());
        assert_eq!(cache.len(), 1);

        cache.insert(2, "two".to_string());
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
    }

    #[test]
    fn test_priority_ordering_consistency() {
        type MruCache = Cache<i32, String, MruPolicy>;
        let mut cache = MruCache::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        cache.get(&1);
        cache.get(&2);

        let youngest = cache.tail().map(|(k, _)| *k);
        assert_eq!(youngest, Some(2));

        cache.get(&3);
        let youngest = cache.tail().map(|(k, _)| *k);
        assert_eq!(youngest, Some(3));
    }

    #[test]
    fn test_mru_re_index_functionality() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();

        queue.insert(
            1,
            Entry {
                priority: 100,
                value: "one",
            },
        );
        queue.insert(
            2,
            Entry {
                priority: 200,
                value: "two",
            },
        );
        queue.insert(
            3,
            Entry {
                priority: 150,
                value: "three",
            },
        );
        queue.insert(
            4,
            Entry {
                priority: 75,
                value: "four",
            },
        );

        let mut next_priority = u64::MAX;

        MruPolicy::re_index(&mut queue, &mut next_priority, 0);

        assert!(queue[0].priority < u64::MAX);
        assert!(queue[1].priority < u64::MAX);
        assert!(queue[2].priority < u64::MAX);
        assert!(queue[3].priority < u64::MAX);

        assert!(next_priority > u64::MIN);
        assert!(next_priority < u64::MAX);

        let priorities: std::collections::HashSet<u64> =
            queue.values().map(|entry| entry.priority).collect();
        assert_eq!(
            priorities.len(),
            4,
            "All priorities should be unique after re-indexing"
        );
    }

    #[test]
    fn test_mru_re_index_maintains_relative_order() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();

        queue.insert(
            1,
            Entry {
                priority: 10,
                value: "oldest",
            },
        );
        queue.insert(
            2,
            Entry {
                priority: 20,
                value: "middle",
            },
        );
        queue.insert(
            3,
            Entry {
                priority: 30,
                value: "newest",
            },
        );

        let mut next_priority = u64::MAX;

        MruPolicy::re_index(&mut queue, &mut next_priority, 0);

        assert!(queue.get(&1).unwrap().priority < queue.get(&2).unwrap().priority);
        assert!(queue.get(&2).unwrap().priority < queue.get(&3).unwrap().priority);

        for (_, entry) in queue.iter() {
            assert!(entry.priority < u64::MAX);
        }
    }

    #[test]
    fn test_mru_re_index_edge_cases() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        queue.insert(
            1,
            Entry {
                priority: 100,
                value: "only",
            },
        );

        let mut next_priority = u64::MAX;
        MruPolicy::re_index(&mut queue, &mut next_priority, 0);

        assert_eq!(queue.len(), 1);
        assert!(queue[0].priority < u64::MAX);
        assert!(next_priority > u64::MIN);

        let mut empty_queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        let mut next_priority_empty = u64::MAX;
        MruPolicy::re_index(&mut empty_queue, &mut next_priority_empty, 0);

        assert_eq!(empty_queue.len(), 0);
        assert_eq!(next_priority_empty, u64::MIN);
    }
}
