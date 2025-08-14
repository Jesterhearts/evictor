use std::hash::Hash;

use indexmap::IndexMap;

use crate::{
    EntryValue,
    Policy,
    RandomState,
};

/// Policy implementation for Least Recently Used (LRU) cache eviction.
///
/// This policy tracks when entries were last accessed and evicts the entry
/// that was accessed furthest in the past when the cache reaches capacity.
/// It implements a priority system where higher priority values indicate
/// more recent access.
///
/// # Implementation Details
///
/// ## Priority System
/// - **Priority Values**: Uses descending u64 values starting from `u64::MAX`
/// - **Recent Access**: Higher priority values = more recently accessed
/// - **Eviction Target**: Entry with lowest priority (oldest access time)
/// - **Priority Assignment**: Each access decrements the global priority
///   counter
///
/// ## Heap Structure
/// The underlying heap maintains the invariant that parent nodes have
/// priorities >= their children, ensuring the least recently used item
/// (lowest priority) is always at index 0 for efficient eviction.
///
/// ## Priority Counter Management
/// - **Start Value**: `u64::MAX` (newest possible priority)
/// - **End Value**: `u64::MIN` (oldest possible priority)
/// - **Overflow Handling**: When counter reaches `u64::MIN`, it wraps to
///   `u64::MAX` and the entire heap is re-indexed to maintain ordering
///
/// # Usage
///
/// This policy is typically used through the [`Lru`] type alias rather than
/// directly:
///
/// ```rust
/// use std::num::NonZeroUsize;
///
/// use evictor::{
///     Cache,
///     LruPolicy,
/// };
///
/// // Direct usage (not recommended)
/// let mut cache: Cache<i32, String, LruPolicy> = Cache::new(NonZeroUsize::new(3).unwrap());
///
/// // Preferred usage via type alias
/// use evictor::Lru;
/// let mut cache: Lru<i32, String> = Lru::new(NonZeroUsize::new(3).unwrap());
/// ```
///
/// # Performance Characteristics
///
/// - **Access Update**: O(log n) - requires heap bubble operation
/// - **Eviction**: O(log n) - removes root and re-heapifies
/// - **Priority Overflow**: O(n) - occurs very rarely (every 2^64 operations)
///
/// [`Lru`]: crate::Lru
#[derive(Debug)]
pub struct LruPolicy;

impl Policy for LruPolicy {
    fn start_value() -> u64 {
        u64::MAX
    }

    fn end_value() -> u64 {
        u64::MIN
    }

    fn assign_update_next_value(cache_next: &mut u64, entry: &mut impl EntryValue) {
        *entry.priority_mut() = *cache_next;
        *cache_next = cache_next.wrapping_sub(1)
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
                if queue[left_idx].priority() > queue[index].priority() {
                    queue.swap_indices(index, left_idx);
                    index = left_idx;
                }
                break;
            }

            let target = if queue[left_idx].priority() > queue[right_idx].priority() {
                left_idx
            } else {
                right_idx
            };

            if queue[target].priority() < queue[index].priority() {
                break;
            }

            queue.swap_indices(index, target);
            index = target;
        }

        index
    }

    fn re_index<T: EntryValue>(
        queue: &mut IndexMap<impl Hash + Eq, T, RandomState>,
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

        let left_idx = index * 2 + 1;
        let right_idx = index * 2 + 2;

        Self::re_index(queue, next_priority, left_idx);
        Self::re_index(queue, next_priority, right_idx);
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use indexmap::IndexMap;

    use super::*;
    use crate::{
        Entry,
        Lru,
    };

    fn assert_heap_property<Key, Value>(lru: &Lru<Key, Value>) {
        for i in 0..lru.queue.len() {
            let left_idx = i * 2 + 1;
            let right_idx = i * 2 + 2;

            if left_idx < lru.queue.len() {
                let parent_age = lru.queue[i].priority;
                let left_child_age = lru.queue[left_idx].priority;
                assert!(
                    parent_age >= left_child_age,
                    "Heap property violated: parent at index {} has age {}, left child at index {} has age {}",
                    i,
                    parent_age,
                    left_idx,
                    left_child_age
                );
            }

            if right_idx < lru.queue.len() {
                let parent_age = lru.queue[i].priority;
                let right_child_age = lru.queue[right_idx].priority;
                assert!(
                    parent_age >= right_child_age,
                    "Heap property violated: parent at index {} has age {}, right child at index {} has age {}",
                    i,
                    parent_age,
                    right_idx,
                    right_child_age
                );
            }
        }
    }

    #[test]
    fn test_lru_policy_start_value() {
        assert_eq!(LruPolicy::start_value(), u64::MAX);
    }

    #[test]
    fn test_lru_policy_end_value() {
        assert_eq!(LruPolicy::end_value(), u64::MIN);
    }

    #[test]
    fn test_assign_update_next_value() {
        let mut cache_next = 100u64;
        let mut entry = Entry {
            priority: 0,
            value: "test",
        };

        LruPolicy::assign_update_next_value(&mut cache_next, &mut entry);

        assert_eq!(entry.priority, 100);
        assert_eq!(cache_next, 99);
    }

    #[test]
    fn test_assign_update_next_value_wrapping() {
        let mut cache_next = u64::MIN;
        let mut entry = Entry {
            priority: 0,
            value: "test",
        };

        LruPolicy::assign_update_next_value(&mut cache_next, &mut entry);

        assert_eq!(entry.priority, u64::MIN);
        assert_eq!(cache_next, u64::MAX);
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

        let result = LruPolicy::heap_bubble(0, &mut queue);
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

        let result = LruPolicy::heap_bubble(1, &mut queue);
        assert_eq!(result, 1);
        assert_eq!(queue[0].priority, 200);
        assert_eq!(queue[1].priority, 100);
    }

    #[test]
    fn test_lru_cache_basic_operations() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());

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
    fn test_lru_cache_eviction_policy() {
        let mut cache = Lru::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.get(&2);

        cache.insert(3, "three".to_string());

        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    fn test_lru_cache_get_updates_priority() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let oldest_before = cache.tail().map(|(k, _)| *k);

        cache.get(&1);

        let oldest_after = cache.tail().map(|(k, _)| *k);
        assert_ne!(oldest_before, oldest_after);
    }

    #[test]
    fn test_lru_cache_peek_does_not_update_priority() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let oldest_before = cache.tail().map(|(k, _)| *k);

        cache.peek(&1);

        let oldest_after = cache.tail().map(|(k, _)| *k);
        assert_eq!(oldest_before, oldest_after);
    }

    #[test]
    fn test_lru_cache_get_or_insert_with() {
        let mut cache = Lru::new(NonZeroUsize::new(2).unwrap());

        let value = cache.get_or_insert_with(1, |_| "one".to_string());
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 1);

        let value = cache.get_or_insert_with(1, |_| "different".to_string());
        assert_eq!(value, "one");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_remove() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let removed = cache.remove(&1);
        assert_eq!(removed, Some("one".to_string()));
        assert!(!cache.contains_key(&1));
        assert_eq!(cache.len(), 1);

        let removed = cache.remove(&1);
        assert_eq!(removed, None);
    }

    #[test]
    fn test_lru_cache_pop() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let (_key, _value) = cache.pop().unwrap();
        assert_eq!(cache.len(), 2);

        let mut empty_cache = Lru::<i32, String>::new(NonZeroUsize::new(1).unwrap());
        assert!(empty_cache.pop().is_none());
    }

    #[test]
    fn test_lru_cache_clear() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lru_cache_retain() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lru_retain_empty_cache() {
        let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(5).unwrap());

        cache.retain(|_, _| true);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.retain(|_, _| false);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lru_retain_all_elements() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lru_retain_none() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lru_retain_with_recency_patterns() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lru_retain_modify_values() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

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
    fn test_lru_retain_single_element() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());
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
    fn test_lru_retain_heap_property_after_removal() {
        let mut cache = Lru::new(NonZeroUsize::new(10).unwrap());

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
    fn test_lru_retain_with_duplicates_and_reinserts() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

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
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert!(cache.contains_key(&5));
        assert!(cache.contains_key(&6));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lru_retain_stress_test() {
        let mut cache = Lru::new(NonZeroUsize::new(100).unwrap());

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
    fn test_lru_retain_capacity_one() {
        let mut cache = Lru::new(NonZeroUsize::new(1).unwrap());
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
    fn test_lru_retain_and_lru_behavior() {
        let mut cache = Lru::new(NonZeroUsize::new(4).unwrap());

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
    fn test_lru_retain_preserves_access_order() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.get(&3);
        cache.get(&1);
        cache.get(&4);

        let oldest_before = cache.tail().map(|(k, _)| *k);

        cache.retain(|_, _| true);

        let oldest_after = cache.tail().map(|(k, _)| *k);

        assert_eq!(oldest_before, oldest_after);
        assert_heap_property(&cache);

        cache.insert(6, "six".to_string());
        cache.insert(7, "seven".to_string());

        assert!(cache.len() <= 5);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lru_cache_extend() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());

        let items = vec![(2, "two".to_string()), (3, "three".to_string())];
        cache.extend(items);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    fn test_lru_cache_from_iterator() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let cache: Lru<i32, String> = items.into_iter().collect();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    fn test_lru_cache_from_iterator_overlapping() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
            (1, "updated_one".to_string()),
        ];

        let cache: Lru<i32, String> = items.into_iter().collect();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lru_cache_priority_wraparound() {
        let mut cache = Lru::new(NonZeroUsize::new(2).unwrap());

        for i in 0..1000000 {
            cache.insert(i, format!("value{}", i));
            if i % 10000 == 0 && i > 0 {
                break;
            }
        }

        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lru_cache_shrink_to_fit() {
        let mut cache = Lru::new(NonZeroUsize::new(10).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.shrink_to_fit();

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
    }

    #[test]
    fn test_lru_cache_mutable_operations() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());

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
    fn test_edge_case_empty_heap_bubble() {
        let mut queue: IndexMap<i32, Entry<&str>, RandomState> = IndexMap::default();
        let result = LruPolicy::heap_bubble(0, &mut queue);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_edge_case_capacity_one() {
        let mut cache = Lru::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one".to_string());
        assert_eq!(cache.len(), 1);

        cache.insert(2, "two".to_string());
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
    }

    #[test]
    fn test_priority_ordering_consistency() {
        let mut cache = Lru::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        cache.get(&3);
        cache.get(&2);

        let oldest = cache.tail().map(|(k, _)| *k);
        assert_eq!(oldest, Some(1));

        cache.get(&1);
        let oldest = cache.tail().map(|(k, _)| *k);
        assert_eq!(oldest, Some(3));
    }

    #[test]
    fn test_capacity() {
        let mut lru = Lru::<i32, i32>::new(std::num::NonZeroUsize::new(5).unwrap());

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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_clear() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());
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
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());
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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_oldest() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());

        assert_eq!(lru.tail(), None);

        lru.insert(1, "one".to_string());
        assert_eq!(lru.tail(), Some((&1, &"one".to_string())));

        lru.insert(2, "two".to_string());
        assert_eq!(lru.tail(), Some((&1, &"one".to_string())));

        lru.get(&1);
        assert_eq!(lru.tail(), Some((&2, &"two".to_string())));
        assert_heap_property(&lru);
    }

    #[test]
    fn test_contains_key() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());

        assert!(!lru.contains_key(&1));

        lru.insert(1, "one".to_string());
        assert!(lru.contains_key(&1));
        assert!(!lru.contains_key(&2));

        lru.remove(&1);
        assert!(!lru.contains_key(&1));
    }

    #[test]
    fn test_get_or_insert_with() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());
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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_get_or_insert_with_mut() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());

        let value = lru.get_or_insert_with_mut(1, |_| "one".to_string());
        *value = "modified".to_string();

        assert_eq!(lru.peek(&1), Some(&"modified".to_string()));

        let value = lru.get_or_insert_with_mut(1, |_| "different".to_string());
        assert_eq!(value, &"modified".to_string());
        assert_heap_property(&lru);
    }

    #[test]
    fn test_insert_and_insert_mut() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());

        let value = lru.insert(1, "one".to_string());
        assert_eq!(value, &"one".to_string());

        let value = lru.insert_mut(2, "two".to_string());
        *value = "modified_two".to_string();

        assert_eq!(lru.peek(&1), Some(&"one".to_string()));
        assert_eq!(lru.peek(&2), Some(&"modified_two".to_string()));

        lru.insert(1, "new_one".to_string());
        assert_eq!(lru.peek(&1), Some(&"new_one".to_string()));
        assert_heap_property(&lru);
    }

    #[test]
    fn test_get_and_get_mut() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());
        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());

        assert_eq!(lru.get(&1), Some(&"one".to_string()));
        assert_eq!(lru.get(&3), None);

        if let Some(value) = lru.get_mut(&2) {
            *value = "modified_two".to_string();
        }

        assert_eq!(lru.peek(&2), Some(&"modified_two".to_string()));
        assert_eq!(lru.get_mut(&3), None);
        assert_heap_property(&lru);
    }

    #[test]
    fn test_pop() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());

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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_remove() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());
        lru.insert(1, "one".to_string());
        lru.insert(2, "two".to_string());

        assert_eq!(lru.remove(&1), Some("one".to_string()));
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.remove(&1), None);
        assert_eq!(lru.remove(&3), None);

        assert_eq!(lru.peek(&1), None);
        assert_eq!(lru.peek(&2), Some(&"two".to_string()));
        assert_heap_property(&lru);
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());

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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_capacity_method() {
        let lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(5).unwrap());
        assert_eq!(lru.capacity(), 5);

        let lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(100).unwrap());
        assert_eq!(lru.capacity(), 100);
        assert_heap_property(&lru);
    }

    #[test]
    fn test_retain() {
        let mut lru = Lru::<i32, i32>::new(std::num::NonZeroUsize::new(5).unwrap());
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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_extend() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(5).unwrap());
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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(10).unwrap());
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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_lru_order_complex() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());

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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_get_or_insert_capacity_behavior() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(2).unwrap());

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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_edge_cases() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(1).unwrap());
        lru.insert(1, "one".to_string());
        assert_eq!(lru.len(), 1);

        lru.insert(2, "two".to_string());
        assert_eq!(lru.len(), 1);
        assert!(!lru.contains_key(&1));
        assert!(lru.contains_key(&2));

        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());
        assert_eq!(lru.get(&1), None);
        assert_eq!(lru.remove(&1), None);
        assert_eq!(lru.pop(), None);
        assert_eq!(lru.tail(), None);
        assert_heap_property(&lru);
    }

    #[test]
    fn test_from_iter_basic() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let lru: Lru<i32, String> = items.into_iter().collect();

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.peek(&1), Some(&"one".to_string()));
        assert_eq!(lru.peek(&2), Some(&"two".to_string()));
        assert_eq!(lru.peek(&3), Some(&"three".to_string()));

        assert_eq!(lru.tail(), Some((&1, &"one".to_string())));
        assert_heap_property(&lru);
    }

    #[test]
    fn test_from_iter_with_multiple_items() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
            (4, "four".to_string()),
            (5, "five".to_string()),
        ];

        let lru: Lru<i32, String> = items.into_iter().collect();

        assert_eq!(lru.len(), 5);
        assert_eq!(lru.peek(&1), Some(&"one".to_string()));
        assert_eq!(lru.peek(&2), Some(&"two".to_string()));
        assert_eq!(lru.peek(&3), Some(&"three".to_string()));
        assert_eq!(lru.peek(&4), Some(&"four".to_string()));
        assert_eq!(lru.peek(&5), Some(&"five".to_string()));

        assert_eq!(lru.tail(), Some((&1, &"one".to_string())));
        assert_heap_property(&lru);
    }

    #[test]
    fn test_from_iter_empty() {
        let items: Vec<(i32, String)> = vec![];
        let lru: Lru<i32, String> = items.into_iter().collect();

        assert!(lru.is_empty());
        assert_eq!(lru.len(), 0);
        assert_eq!(lru.tail(), None);
    }

    #[test]
    fn test_from_iter_single_item() {
        let items = vec![(42, "answer".to_string())];
        let lru: Lru<i32, String> = items.into_iter().collect();

        assert_eq!(lru.len(), 1);
        assert_eq!(lru.peek(&42), Some(&"answer".to_string()));
        assert_eq!(lru.tail(), Some((&42, &"answer".to_string())));
    }

    #[test]
    fn test_from_iter_lru_order() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let mut lru: Lru<i32, String> = items.into_iter().collect();

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
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(3).unwrap());

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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_age_counter_stress() {
        let mut lru = Lru::<i32, i32>::new(std::num::NonZeroUsize::new(100).unwrap());
        lru.next_priority = 50;

        for i in 0..200 {
            lru.insert(i, i * 2);
            if i % 20 == 0 {
                lru.get(&(i / 2));
            }
        }

        assert_eq!(lru.len(), 100);
        assert_heap_property(&lru);

        let mut prev_age = u64::MAX;
        while let Some((_, entry)) = lru.pop_internal() {
            assert!(entry.priority <= prev_age, "Age order violated");
            prev_age = entry.priority;
        }
    }

    #[test]
    fn test_remove_from_different_positions() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            lru.insert(i, format!("value_{}", i));
        }

        lru.get(&3);
        lru.get(&1);
        lru.get(&4);

        assert_eq!(lru.remove(&3), Some("value_3".to_string()));
        assert_eq!(lru.len(), 4);

        let oldest = *lru.tail().unwrap().0;
        assert_eq!(lru.remove(&oldest), Some(format!("value_{}", oldest)));
        assert_eq!(lru.len(), 3);

        let mut ages = vec![];
        while let Some((_, entry)) = lru.pop_internal() {
            ages.push(entry.priority);
        }

        for window in ages.windows(2) {
            assert!(window[0] >= window[1], "Heap property violated");
        }
    }

    #[test]
    fn test_extend_with_overlapping_keys() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(4).unwrap());

        lru.insert(1, "original_1".to_string());
        lru.insert(2, "original_2".to_string());

        let items = vec![
            (2, "updated_2".to_string()),
            (2, "updated_again_2".to_string()),
            (3, "new_3".to_string()),
            (4, "new_4".to_string()),
            (5, "new_5".to_string()),
        ];

        lru.extend(items);

        assert_eq!(lru.len(), 4);
        assert_eq!(lru.peek(&2), Some(&"updated_again_2".to_string()));
        assert_eq!(lru.peek(&3), Some(&"new_3".to_string()));
        assert_eq!(lru.peek(&4), Some(&"new_4".to_string()));
        assert_eq!(lru.peek(&5), Some(&"new_5".to_string()));

        assert!(!lru.contains_key(&1));
        assert_heap_property(&lru);
    }

    #[test]
    fn test_alternating_access_patterns() {
        let mut lru = Lru::<i32, String>::new(std::num::NonZeroUsize::new(4).unwrap());

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
        assert_heap_property(&lru);
    }

    #[test]
    fn test_large_capacity_operations() {
        let mut lru = Lru::<usize, usize>::new(std::num::NonZeroUsize::new(1000).unwrap());

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

        assert_heap_property(&lru);
        let mut prev_age = u64::MAX;
        while let Some((_, entry)) = lru.pop_internal() {
            assert!(
                entry.priority <= prev_age,
                "Heap order violated in large cache"
            );
            prev_age = entry.priority;
        }
    }

    #[test]
    fn test_lru_heap_corruption_and_heapify() {
        let mut lru = Lru::<i32, i32>::new(NonZeroUsize::new(7).unwrap());

        for i in 0..7 {
            lru.insert(i, i * 10);
        }

        {
            let queue = &mut lru.queue;
            queue[1].priority = 9999;
            queue[3].priority = 1;
            queue[5].priority = 8888;
        }

        lru.heapify();

        assert_heap_property(&lru);
    }
}
