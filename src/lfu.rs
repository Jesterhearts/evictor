use std::hash::Hash;

use indexmap::IndexMap;

use crate::{
    EntryValue,
    Policy,
    RandomState,
};

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

        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert!(!cache.contains_key(&1));
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
