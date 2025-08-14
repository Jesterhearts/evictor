use std::hash::Hash;

use crate::{
    EntryValue,
    Policy,
    RandomState,
};

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
        while index > 0 {
            let left_idx = (index / 2).saturating_sub(1);
            let right_idx = index / 2;

            if right_idx == 0 {
                if queue[0].priority() < queue[index].priority() {
                    queue.swap_indices(right_idx, index);
                }
                break;
            }

            let target = if queue[left_idx].priority() > queue[right_idx].priority() {
                left_idx
            } else {
                right_idx
            };

            queue.swap_indices(target, index);
            index = target;
        }

        0
    }

    fn re_index<T: EntryValue>(
        queue: &mut indexmap::IndexMap<impl Hash + Eq, T, RandomState>,
        next_priority: &mut u64,
        index: usize,
    ) {
        if *next_priority == Self::end_value() {
            *next_priority = Self::start_value();
        }

        Self::assign_update_next_value(next_priority, &mut queue[index]);

        if index == 0 {
            return;
        }

        let left_index = (index / 2).saturating_sub(1);
        let right_index = index / 2;

        if left_index != right_index {
            Self::re_index(queue, next_priority, left_index);
        }
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
    fn test_mru_policy_start_value() {
        assert_eq!(MruPolicy::start_value(), u64::MIN);
    }

    #[test]
    fn test_mru_policy_end_value() {
        assert_eq!(MruPolicy::end_value(), u64::MAX);
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
        assert_eq!(result, 0);
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
}
