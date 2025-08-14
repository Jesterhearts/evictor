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

        loop {
            let left_index = index * 2 + 1;
            let right_index = index * 2 + 2;

            if left_index >= queue.len() {
                break;
            }

            if right_index >= queue.len() {
                if queue[left_index].priority() > queue[index].priority() {
                    queue.swap_indices(index, left_index);
                    index = left_index;
                }
                break;
            }

            let target = if queue[left_index].priority() > queue[right_index].priority() {
                left_index
            } else {
                right_index
            };

            if queue[target].priority() <= queue[index].priority() {
                break;
            }
            queue.swap_indices(index, target);
            index = target;
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
        Mru,
    };

    fn assert_heap_property<Key, Value>(mru: &Mru<Key, Value>) {
        for i in 0..mru.queue.len() {
            let left_idx = i * 2 + 1;
            let right_idx = i * 2 + 2;

            if left_idx < mru.queue.len() {
                let parent_priority = mru.queue[i].priority;
                let left_child_priority = mru.queue[left_idx].priority;
                assert!(
                    parent_priority >= left_child_priority,
                    "Heap property violated: parent at index {} has priority {}, left child at index {} has priority {}",
                    i,
                    parent_priority,
                    left_idx,
                    left_child_priority
                );
            }

            if right_idx < mru.queue.len() {
                let parent_priority = mru.queue[i].priority;
                let right_child_priority = mru.queue[right_idx].priority;
                assert!(
                    parent_priority >= right_child_priority,
                    "Heap property violated: parent at index {} has priority {}, right child at index {} has priority {}",
                    i,
                    parent_priority,
                    right_idx,
                    right_child_priority
                );
            }
        }
    }

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
    fn test_mru_cache_basic_operations() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());

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
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_eviction_policy() {
        let mut cache = Mru::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.get(&1);

        cache.insert(3, "three".to_string());

        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_get_updates_priority() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let oldest_before = cache.tail().map(|(k, _)| *k);

        cache.get(&1);

        let oldest_after = cache.tail().map(|(k, _)| *k);
        assert_ne!(oldest_before, oldest_after);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_peek_does_not_update_priority() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let oldest_before = cache.tail().map(|(k, _)| *k);

        cache.peek(&1);

        let oldest_after = cache.tail().map(|(k, _)| *k);
        assert_eq!(oldest_before, oldest_after);
    }

    #[test]
    fn test_mru_cache_get_or_insert_with() {
        let mut cache = Mru::new(NonZeroUsize::new(2).unwrap());

        let value = cache.get_or_insert_with(1, |_| "one".to_string());
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 1);

        let value = cache.get_or_insert_with(1, |_| "different".to_string());
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_mru_cache_remove() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let removed = cache.remove(&1);
        assert_eq!(removed, Some("one".to_string()));
        assert!(!cache.contains_key(&1));
        assert_eq!(cache.len(), 2);
        let youngest = cache.tail().map(|(k, _)| *k);
        assert_eq!(youngest, Some(3));
        assert_heap_property(&cache);

        let removed = cache.remove(&1);
        assert_eq!(removed, None);
    }

    #[test]
    fn test_mru_cache_pop() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        cache.get(&1);
        cache.get(&3);
        let (_key, _value) = cache.pop().unwrap();
        assert_eq!(cache.len(), 2);
        assert_heap_property(&cache);

        let mut empty_cache = Mru::<i32, String>::new(NonZeroUsize::new(1).unwrap());
        assert!(empty_cache.pop().is_none());
    }

    #[test]
    fn test_mru_cache_clear() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_mru_cache_retain() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

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
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_empty_cache() {
        let mut cache = Mru::<i32, String>::new(NonZeroUsize::new(5).unwrap());

        cache.retain(|_, _| true);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.retain(|_, _| false);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_all_elements() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.retain(|_, _| true);

        assert_eq!(cache.len(), 5);
        for i in 1..=5 {
            assert!(cache.contains_key(&i));
        }
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_none() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.retain(|_, _| false);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        for i in 1..=5 {
            assert!(!cache.contains_key(&i));
        }
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_with_recency_patterns() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());
        cache.insert(4, "four".to_string());
        cache.insert(5, "five".to_string());

        cache.get(&1);
        cache.get(&3);
        cache.get(&5);

        cache.retain(|&k, _| k == 1 || k == 3 || k == 5);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(!cache.contains_key(&4));
        assert!(cache.contains_key(&5));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_modify_values() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);
        cache.insert(4, 40);
        cache.insert(5, 50);

        cache.retain(|&k, v| {
            if k % 2 == 0 {
                *v *= 2;
                true
            } else {
                false
            }
        });

        assert_eq!(cache.len(), 2);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert!(!cache.contains_key(&5));

        assert_eq!(cache.peek(&2), Some(&40));
        assert_eq!(cache.peek(&4), Some(&80));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_single_element() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());
        cache.insert(42, "answer".to_string());

        cache.retain(|&k, _| k == 42);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&42));
        assert_heap_property(&cache);

        cache.retain(|&k, _| k != 42);
        assert!(cache.is_empty());
        assert!(!cache.contains_key(&42));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_heap_property_after_removal() {
        let mut cache = Mru::new(NonZeroUsize::new(10).unwrap());

        for i in 1..=10 {
            cache.insert(i, format!("value_{}", i));
        }

        for i in &[1, 3, 5, 7, 9] {
            cache.get(i);
        }

        cache.retain(|&k, _| k > 5);

        assert_eq!(cache.len(), 5);
        for i in 1..=5 {
            assert!(!cache.contains_key(&i));
        }
        for i in 6..=10 {
            assert!(cache.contains_key(&i));
        }
        assert_heap_property(&cache);

        cache.insert(11, "eleven".to_string());
        cache.get(&7);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_with_duplicates_and_reinserts() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, i);
        }

        cache.get(&1);
        cache.get(&3);
        cache.get(&5);

        cache.retain(|&k, _| k % 2 == 1);
        assert_eq!(cache.len(), 3);
        assert_heap_property(&cache);

        cache.insert(2, 200);
        cache.insert(4, 400);
        cache.insert(6, 600);

        assert_eq!(cache.len(), 5);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&5));
        assert!(cache.contains_key(&6));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_stress_test() {
        let mut cache = Mru::new(NonZeroUsize::new(100).unwrap());

        for i in 0..100 {
            cache.insert(i, format!("value_{}", i));
        }

        for i in 0..50 {
            cache.get(&(i % 25));
        }

        cache.retain(|&k, v| {
            let should_retain = k % 3 == 0 || k % 7 == 0;
            if should_retain {
                *v = format!("retained_{}", k);
            }
            should_retain
        });

        assert_heap_property(&cache);

        for i in 0..100 {
            if i % 3 == 0 || i % 7 == 0 {
                if cache.contains_key(&i) {
                    assert_eq!(cache.peek(&i), Some(&format!("retained_{}", i)));
                }
            } else {
                assert!(!cache.contains_key(&i));
            }
        }
    }

    #[test]
    fn test_mru_retain_capacity_one() {
        let mut cache = Mru::new(NonZeroUsize::new(1).unwrap());
        cache.insert(1, "one".to_string());

        cache.retain(|_, _| true);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&1));
        assert_heap_property(&cache);

        cache.retain(|_, _| false);
        assert!(cache.is_empty());
        assert_heap_property(&cache);

        cache.insert(2, "two".to_string());
        cache.retain(|&k, _| k > 1);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&2));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_and_mru_behavior() {
        let mut cache = Mru::new(NonZeroUsize::new(4).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());
        cache.insert(4, "four".to_string());

        cache.get(&2);
        cache.get(&1);
        cache.get(&4);

        cache.retain(|&k, _| k == 3 || k == 2);

        assert_eq!(cache.len(), 2);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(!cache.contains_key(&4));
        assert_heap_property(&cache);

        cache.insert(5, "five".to_string());
        cache.insert(6, "six".to_string());
        cache.insert(7, "seven".to_string());

        assert_eq!(cache.len(), 4);
        assert!(cache.contains_key(&2) || cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_preserves_access_order() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.get(&3);
        cache.get(&1);
        cache.get(&4);

        let most_recent_before = cache.tail().map(|(k, _)| *k);

        cache.retain(|_, _| true);

        let most_recent_after = cache.tail().map(|(k, _)| *k);

        assert_eq!(most_recent_before, most_recent_after);
        assert_heap_property(&cache);

        cache.insert(6, "six".to_string());
        cache.insert(7, "seven".to_string());

        assert!(cache.len() <= 5);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_retain_vs_lru_behavior() {
        let mut mru_cache = Mru::new(NonZeroUsize::new(3).unwrap());
        let mut lru_cache = crate::Lru::new(NonZeroUsize::new(3).unwrap());

        for i in 1..=3 {
            mru_cache.insert(i, format!("value_{}", i));
            lru_cache.insert(i, format!("value_{}", i));
        }

        mru_cache.get(&1);
        lru_cache.get(&1);

        mru_cache.retain(|&k, _| k == 1);
        lru_cache.retain(|&k, _| k == 1);

        assert_eq!(mru_cache.len(), 1);
        assert_eq!(lru_cache.len(), 1);
        assert!(mru_cache.contains_key(&1));
        assert!(lru_cache.contains_key(&1));

        mru_cache.insert(4, "four".to_string());
        mru_cache.insert(5, "five".to_string());
        lru_cache.insert(4, "four".to_string());
        lru_cache.insert(5, "five".to_string());

        mru_cache.get(&1);
        lru_cache.get(&1);

        mru_cache.insert(6, "six".to_string());
        lru_cache.insert(6, "six".to_string());

        assert!(!mru_cache.contains_key(&1));
        assert!(lru_cache.contains_key(&1));

        assert_heap_property(&mru_cache);
    }

    #[test]
    fn test_mru_cache_extend() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());

        let items = vec![(2, "two".to_string()), (3, "three".to_string())];
        cache.extend(items);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_from_iterator() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let cache: Mru<i32, String> = items.into_iter().collect();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_priority_wraparound() {
        let mut cache = Mru::new(NonZeroUsize::new(2).unwrap());

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
        let mut cache = Mru::new(NonZeroUsize::new(10).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.shrink_to_fit();

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_mutable_operations() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());

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
        assert_heap_property(&cache);
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
        assert_heap_property(&mru_cache);
    }

    #[test]
    fn test_edge_case_empty_heap_bubble() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        let result = MruPolicy::heap_bubble(0, &mut queue);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_edge_case_capacity_one() {
        let mut cache = Mru::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one".to_string());
        assert_eq!(cache.len(), 1);

        cache.insert(2, "two".to_string());
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
    }

    #[test]
    fn test_priority_ordering_consistency() {
        let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());

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
        assert_heap_property(&cache);
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

    #[test]
    fn test_mru_cache_insert_same_key_multiple_times() {
        let mut cache = Mru::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "first");
        cache.insert(2, "second");

        cache.insert(1, "updated_first");
        assert_eq!(cache.peek(&1), Some(&"updated_first"));
        assert_eq!(cache.len(), 2);

        cache.insert(3, "third");

        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_alternating_access_pattern() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        for _ in 0..10 {
            cache.get(&1);
            cache.get(&2);
        }

        cache.insert(4, "four");

        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_get_or_insert_with_eviction() {
        let mut cache = Mru::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");

        cache.get(&1);

        let value = cache.get_or_insert_with(3, |_| "three");
        assert_eq!(*value, "three");
        assert_eq!(cache.len(), 2);

        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_cache_interleaved_operations() {
        let mut cache = Mru::new(NonZeroUsize::new(4).unwrap());

        cache.insert(1, 10);
        cache.insert(2, 20);
        assert_eq!(cache.get(&1), Some(&10));
        cache.insert(3, 30);

        if let Some(val) = cache.get_mut(&2) {
            *val = 200;
        }

        cache.insert(4, 40);
        cache.insert(5, 50);

        assert_eq!(cache.len(), 4);
        assert_heap_property(&cache);

        cache.retain(|&k, _| k % 2 == 1);

        let items: Vec<_> = (0..10).map(|i| (i + 10, (i + 10) * 10)).collect();
        cache.extend(items);

        assert!(cache.len() <= 4);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_mru_heap_corruption_and_heapify() {
        let mut mru = Mru::<i32, i32>::new(NonZeroUsize::new(7).unwrap());

        for i in 0..7 {
            mru.insert(i, i * 10);
        }

        {
            let queue = &mut mru.queue;
            queue[0].priority = 5;
            queue[2].priority = 999;
            queue[4].priority = 1;
        }

        mru.heapify();

        assert_heap_property(&mru);
    }
}
