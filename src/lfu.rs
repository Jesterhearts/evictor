use std::hash::Hash;

use indexmap::IndexMap;

use crate::{
    EntryValue,
    Policy,
    RandomState,
};

/// Policy implementation for Least Frequently Used (LFU) cache eviction.
///
/// This policy tracks how frequently entries have been accessed and evicts
/// the entry that has been accessed the fewest times when the cache reaches
/// capacity. Unlike LRU/MRU which focus on recency, LFU focuses on frequency
/// of access, making it ideal for workloads with strong frequency-based
/// locality patterns.
///
/// # Implementation Details
///
/// ## Frequency Tracking System
/// - **Priority Values**: Each entry maintains a frequency counter starting at
///   0
/// - **Frequency Updates**: Counter increments on every access (get, insert,
///   etc.)
/// - **Eviction Target**: Entry with lowest frequency count
/// - **Tie Breaking**: When frequencies are equal, heap ordering determines
///   eviction
///
/// ## Heap Structure
/// The underlying heap maintains a min-heap invariant where parent nodes have
/// frequencies <= their children, ensuring the least frequently used item
/// (lowest frequency) is always at index 0 for efficient eviction.
///
/// ## Priority Counter Management
/// - **Start Value**: `0` (no global counter used - each entry tracks its own
///   frequency)
/// - **End Value**: `1` (used as a sentinel, re_index is not supported)
/// - **Overflow Handling**: Uses `saturating_add` to prevent frequency overflow
/// - **No Re-indexing**: LFU doesn't support global re-indexing since
///   frequencies are entry-specific rather than based on a global timestamp
///
/// # Usage
///
/// This policy is typically used through the [`Lfu`] type alias rather than
/// directly:
///
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::{
///     Cache,
///     LfuPolicy,
/// };
///
/// // Direct usage (not recommended)
/// let mut cache: Cache<i32, String, LfuPolicy> = Cache::new(NonZeroUsize::new(3).unwrap());
///
/// // Preferred usage via type alias
/// use evictor::Lfu;
/// let mut cache: Lfu<i32, String> = Lfu::new(NonZeroUsize::new(3).unwrap());
///
/// cache.insert(1, "first".to_string()); // frequency: 0
/// cache.insert(2, "second".to_string()); // frequency: 0
///
/// // Build up different frequency counts
/// cache.get(&1); // frequency: 1
/// cache.get(&1); // frequency: 2
/// cache.get(&2); // frequency: 1
///
/// // Insert when full - evicts the least frequently used
/// cache.insert(3, "third".to_string());
/// // Key 1 has frequency 2, key 2 has frequency 1, key 3 has frequency 0
/// // One of the entries with lowest frequency would be evicted
/// ```
///
/// # Performance Characteristics
///
/// - **Access Update**: O(log n) - requires heap bubble operation
/// - **Eviction**: O(log n) - removes root and re-heapifies
/// - **Frequency Overflow**: Handled via saturation (very rare, requires 2^64
///   accesses)
///
/// [`Lfu`]: crate::Lfu
#[derive(Debug)]
pub struct LfuPolicy;
impl Policy for LfuPolicy {
    fn start_value() -> u64 {
        0
    }

    fn end_value() -> u64 {
        1
    }

    fn assign_update_next_value(_cache_next: &mut u64, entry: &mut impl EntryValue) {
        *entry.priority_mut() = entry.priority().saturating_add(1);
    }

    fn heap_bubble<T: EntryValue>(
        mut index: usize,
        queue: &mut IndexMap<impl Hash + Eq, T, RandomState>,
    ) -> usize {
        loop {
            let left_idx = index * 2 + 1;
            let right_idx = index * 2 + 2;

            if left_idx >= queue.len() {
                break;
            }

            if right_idx >= queue.len() {
                if queue[left_idx].priority() < queue[index].priority() {
                    queue.swap_indices(index, left_idx);
                    index = left_idx;
                }
                break;
            }

            let target = if queue[left_idx].priority() < queue[right_idx].priority() {
                left_idx
            } else {
                right_idx
            };

            if queue[target].priority() > queue[index].priority() {
                break;
            }

            queue.swap_indices(index, target);
            index = target;
        }

        let mut parent_index = index.saturating_sub(1) / 2;
        while index > 0 && queue[index].priority() < queue[parent_index].priority() {
            queue.swap_indices(index, parent_index);
            index = parent_index;
            parent_index = index.saturating_sub(1) / 2;
        }

        index
    }

    fn re_index<T: EntryValue>(
        _queue: &mut IndexMap<impl Hash + Eq, T, RandomState>,
        _next_priority: &mut u64,
        _index: usize,
    ) {
        unreachable!(
            "LFU does not support re-indexing, as it does not update the cache next value."
        );
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use indexmap::IndexMap;

    use super::*;
    use crate::{
        Entry,
        Lfu,
    };

    fn assert_heap_property<Key, Value>(lfu: &Lfu<Key, Value>) {
        for i in 0..lfu.queue.len() {
            let left_idx = i * 2 + 1;
            let right_idx = i * 2 + 2;

            if left_idx < lfu.queue.len() {
                let parent_priority = lfu.queue[i].priority;
                let left_child_priority = lfu.queue[left_idx].priority;
                assert!(
                    parent_priority <= left_child_priority,
                    "Min-heap property violated: parent at index {} has priority {}, left child at index {} has priority {}",
                    i,
                    parent_priority,
                    left_idx,
                    left_child_priority
                );
            }

            if right_idx < lfu.queue.len() {
                let parent_priority = lfu.queue[i].priority;
                let right_child_priority = lfu.queue[right_idx].priority;
                assert!(
                    parent_priority <= right_child_priority,
                    "Min-heap property violated: parent at index {} has priority {}, right child at index {} has priority {}",
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
            priority: 5,
            value: "test",
        };

        LfuPolicy::assign_update_next_value(&mut cache_next, &mut entry);

        assert_eq!(entry.priority, 6);
        assert_eq!(cache_next, 100);
    }

    #[test]
    fn test_assign_update_next_value_saturation() {
        let mut cache_next = 100u64;
        let mut entry = Entry {
            priority: u64::MAX,
            value: "test",
        };

        LfuPolicy::assign_update_next_value(&mut cache_next, &mut entry);

        assert_eq!(entry.priority, u64::MAX);
        assert_eq!(cache_next, 100);
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

        let result = LfuPolicy::heap_bubble(0, &mut queue);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_heap_bubble_two_elements_no_swap() {
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

        let result = LfuPolicy::heap_bubble(1, &mut queue);
        assert_eq!(result, 1);
        assert_eq!(queue[0].priority, 100);
        assert_eq!(queue[1].priority, 200);
    }

    #[test]
    fn test_heap_bubble_two_elements_with_swap() {
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

        let result = LfuPolicy::heap_bubble(0, &mut queue);
        assert_eq!(result, 1);
        assert_eq!(queue[0].priority, 100);
        assert_eq!(queue[1].priority, 200);
    }

    #[test]
    fn test_heap_bubble_multiple_elements() {
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
        queue.insert(
            3,
            Entry {
                priority: 75,
                value: "three",
            },
        );
        queue.insert(
            4,
            Entry {
                priority: 300,
                value: "four",
            },
        );

        let result = LfuPolicy::heap_bubble(0, &mut queue);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_lfu_cache_basic_operations() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

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
    fn test_lfu_cache_eviction_policy() {
        let mut cache = Lfu::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.get(&1);
        cache.get(&1);

        cache.insert(3, "three".to_string());

        assert!(cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_get_updates_priority() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let least_frequent_before = cache.tail().map(|(k, _)| *k);

        cache.get(&1);

        let least_frequent_after = cache.tail().map(|(k, _)| *k);
        assert_ne!(least_frequent_before, least_frequent_after);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_peek_does_not_update_priority() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let least_frequent_before = cache.tail().map(|(k, _)| *k);

        cache.peek(&1);

        let least_frequent_after = cache.tail().map(|(k, _)| *k);
        assert_eq!(least_frequent_before, least_frequent_after);
    }

    #[test]
    fn test_lfu_cache_get_or_insert_with() {
        let mut cache = Lfu::new(NonZeroUsize::new(2).unwrap());

        let value = cache.get_or_insert_with(1, |_| "one".to_string());
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 1);

        let value = cache.get_or_insert_with(1, |_| "different".to_string());
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lfu_cache_remove() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let removed = cache.remove(&1);
        assert_eq!(removed, Some("one".to_string()));
        assert!(!cache.contains_key(&1));
        assert_eq!(cache.len(), 2);
        assert_heap_property(&cache);

        let removed = cache.remove(&1);
        assert_eq!(removed, None);
    }

    #[test]
    fn test_lfu_cache_pop() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let (_key, _value) = cache.pop().unwrap();
        assert_eq!(cache.len(), 2);
        assert_heap_property(&cache);

        let mut empty_cache = Lfu::<i32, String>::new(NonZeroUsize::new(1).unwrap());
        assert!(empty_cache.pop().is_none());
    }

    #[test]
    fn test_lfu_cache_clear() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lfu_cache_retain() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lfu_retain_empty_cache() {
        let mut cache = Lfu::<i32, String>::new(NonZeroUsize::new(5).unwrap());

        cache.retain(|_, _| true);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.retain(|_, _| false);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lfu_retain_all_elements() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lfu_retain_none() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.retain(|_, _| false);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        for i in 1..=5 {
            assert!(!cache.contains_key(&i));
        }
    }

    #[test]
    fn test_lfu_retain_with_frequency_changes() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());
        cache.insert(4, "four".to_string());
        cache.insert(5, "five".to_string());

        cache.get(&1);
        cache.get(&1);
        cache.get(&1);
        cache.get(&2);
        cache.get(&2);
        cache.get(&3);

        cache.retain(|&k, _| k == 1 || k == 2);

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(!cache.contains_key(&4));
        assert!(!cache.contains_key(&5));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_retain_modify_values() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lfu_retain_single_element() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());
        cache.insert(42, "answer".to_string());

        cache.retain(|&k, _| k == 42);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&42));

        cache.retain(|&k, _| k != 42);
        assert!(cache.is_empty());
        assert!(!cache.contains_key(&42));
    }

    #[test]
    fn test_lfu_retain_heap_property_after_removal() {
        let mut cache = Lfu::new(NonZeroUsize::new(10).unwrap());

        for i in 1..=10 {
            cache.insert(i, format!("value_{}", i));
        }

        for _ in 0..5 {
            cache.get(&1);
        }
        for _ in 0..3 {
            cache.get(&3);
        }
        for _ in 0..7 {
            cache.get(&7);
        }
        for _ in 0..2 {
            cache.get(&2);
            cache.get(&9);
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
    fn test_lfu_retain_with_duplicates_and_reinserts() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, i);
        }

        cache.get(&1);
        cache.get(&3);
        cache.get(&5);

        cache.retain(|&k, _| k % 2 == 1);
        assert_eq!(cache.len(), 3);

        cache.insert(2, 200);
        cache.insert(4, 400);
        cache.insert(6, 600);

        assert_eq!(cache.len(), 5);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert!(cache.contains_key(&5));
        assert!(cache.contains_key(&6));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_retain_stress_test() {
        let mut cache = Lfu::new(NonZeroUsize::new(100).unwrap());

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
    fn test_lfu_retain_capacity_one() {
        let mut cache = Lfu::new(NonZeroUsize::new(1).unwrap());
        cache.insert(1, "one".to_string());

        cache.retain(|_, _| true);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&1));

        cache.retain(|_, _| false);
        assert!(cache.is_empty());

        cache.insert(2, "two".to_string());
        cache.retain(|&k, _| k > 1);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&2));
    }

    #[test]
    fn test_lfu_cache_extend() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lfu_cache_from_iterator() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let cache: Lfu<i32, String> = items.into_iter().collect();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_from_iter_overlapping() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
            (3, "three_new".to_string()),
        ];

        let cache: Lfu<i32, String> = items.into_iter().collect();

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_shrink_to_fit() {
        let mut cache = Lfu::new(NonZeroUsize::new(10).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.shrink_to_fit();

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_mutable_operations() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

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
    fn test_lfu_vs_lru_behavior_difference() {
        use crate::{
            Lfu,
            Lru,
        };

        let mut lfu_cache: Lfu<i32, String> = Lfu::new(NonZeroUsize::new(2).unwrap());
        let mut lru_cache: Lru<i32, String> = Lru::new(NonZeroUsize::new(2).unwrap());

        lfu_cache.insert(1, "one".to_string());
        lfu_cache.insert(2, "two".to_string());
        lru_cache.insert(1, "one".to_string());
        lru_cache.insert(2, "two".to_string());

        lfu_cache.get(&1);
        lfu_cache.get(&1);
        lru_cache.get(&1);

        lfu_cache.insert(3, "three".to_string());
        lru_cache.insert(3, "three".to_string());

        assert!(lfu_cache.contains_key(&1));
        assert!(lru_cache.contains_key(&1));

        assert!(!lfu_cache.contains_key(&2));
        assert!(!lru_cache.contains_key(&2));

        assert!(lfu_cache.contains_key(&3));
        assert!(lru_cache.contains_key(&3));
        assert_heap_property(&lfu_cache);
    }

    #[test]
    fn test_lfu_vs_mru_behavior_difference() {
        use crate::{
            Lfu,
            Mru,
        };

        let mut lfu_cache: Lfu<i32, String> = Lfu::new(NonZeroUsize::new(2).unwrap());
        let mut mru_cache: Mru<i32, String> = Mru::new(NonZeroUsize::new(2).unwrap());

        lfu_cache.insert(1, "one".to_string());
        lfu_cache.insert(2, "two".to_string());
        mru_cache.insert(1, "one".to_string());
        mru_cache.insert(2, "two".to_string());

        lfu_cache.get(&1);
        lfu_cache.get(&1);
        mru_cache.get(&1);

        lfu_cache.insert(3, "three".to_string());
        mru_cache.insert(3, "three".to_string());

        assert!(lfu_cache.contains_key(&1));
        assert!(!mru_cache.contains_key(&1));

        assert!(!lfu_cache.contains_key(&2));
        assert!(mru_cache.contains_key(&2));

        assert!(lfu_cache.contains_key(&3));
        assert!(mru_cache.contains_key(&3));
        assert_heap_property(&lfu_cache);
    }

    #[test]
    fn test_edge_case_empty_heap_bubble() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        let result = LfuPolicy::heap_bubble(0, &mut queue);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_edge_case_capacity_one() {
        let mut cache = Lfu::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one".to_string());
        assert_eq!(cache.len(), 1);

        cache.insert(2, "two".to_string());
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
    }

    #[test]
    fn test_frequency_ordering_consistency() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        cache.get(&1);
        cache.get(&1);
        cache.get(&2);

        let least_frequent = cache.tail().map(|(k, _)| *k);
        assert_eq!(least_frequent, Some(3));

        cache.get(&3);
        cache.get(&3);
        cache.get(&3);

        let least_frequent = cache.tail().map(|(k, _)| *k);
        assert_eq!(least_frequent, Some(2));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_frequency_tracking() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.get(&1);
        cache.get(&2);
        cache.get(&2);

        cache.insert(3, "three".to_string());
        cache.insert(4, "four".to_string());

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&4));
        assert!(!cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_complex_frequency_scenario() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        for _ in 0..5 {
            cache.get(&1);
        }
        for _ in 0..3 {
            cache.get(&2);
        }

        cache.insert(4, "four".to_string());

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(cache.contains_key(&4));

        cache.insert(5, "five".to_string());

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&4));
        assert!(cache.contains_key(&5));
        assert_heap_property(&cache);
    }

    #[test]
    #[should_panic(
        expected = "LFU does not support re-indexing, as it does not update the cache next value."
    )]
    fn test_re_index_panics() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        let mut next_priority = 0u64;
        LfuPolicy::re_index(&mut queue, &mut next_priority, 0);
    }
}
