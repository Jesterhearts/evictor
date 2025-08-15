//! Least Frequently Used (LFU) cache implementation using a min-heap.
//!
//! This module provides the LFU eviction policy, which removes the least
//! frequently accessed entry when the cache reaches capacity. Uses a heap
//! structure to efficiently track frequency-based priorities.

use std::hash::Hash;

use indexmap::IndexMap;

use crate::{
    EntryValue,
    Metadata,
    Policy,
    RandomState,
    private,
};

/// LFU cache eviction policy implementation using heap structure.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct LfuPolicy;
impl private::Sealed for LfuPolicy {}

/// LFU cache entry containing frequency counter and value.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct LfuEntry<T> {
    uses: u64,
    value: T,
}
impl<T> private::Sealed for LfuEntry<T> {}

impl<T> EntryValue<T> for LfuEntry<T> {
    fn new(value: T) -> Self {
        LfuEntry { uses: 0, value }
    }

    fn prepare_for_reinsert(&mut self) {
        self.uses = self.uses.saturating_sub(1)
    }

    fn into_value(self) -> T {
        self.value
    }

    fn value(&self) -> &T {
        &self.value
    }

    fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

/// Metadata for LFU policy (minimal - heap maintains its own order).
#[derive(Debug, Default, Clone, Copy)]
#[doc(hidden)]
pub struct LfuMetadata;
impl private::Sealed for LfuMetadata {}

impl Metadata for LfuMetadata {
    fn tail_index(&self) -> usize {
        0
    }
}

impl<T> Policy<T> for LfuPolicy {
    type EntryType = LfuEntry<T>;
    type MetadataType = LfuMetadata;

    fn touch_entry(
        index: usize,
        _metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<impl Hash + Eq, Self::EntryType, RandomState>,
    ) -> usize {
        if index >= queue.len() {
            return index;
        }

        queue[index].uses = queue[index].uses.saturating_add(1);
        heap_bubble(index, queue)
    }

    fn swap_remove_entry<K: Hash + Eq>(
        index: usize,
        _metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<K, Self::EntryType, RandomState>,
    ) -> (usize, Option<(K, Self::EntryType)>) {
        let result = queue.swap_remove_index(index);

        if index >= queue.len() {
            (index, result)
        } else {
            (heap_bubble(index, queue), result)
        }
    }
}

/// Maintains heap property by bubbling an entry to its correct position.
///
/// This function ensures the min-heap invariant is maintained after
/// frequency updates or structural changes.
fn heap_bubble<T>(
    mut index: usize,
    queue: &mut IndexMap<impl Hash + Eq, LfuEntry<T>, ahash::RandomState>,
) -> usize {
    loop {
        let left_idx = index * 2 + 1;
        let right_idx = index * 2 + 2;

        if left_idx >= queue.len() {
            break;
        }

        if right_idx >= queue.len() {
            if queue[left_idx].uses < queue[index].uses {
                queue.swap_indices(index, left_idx);
                index = left_idx;
            }
            break;
        }

        let target = if queue[left_idx].uses < queue[right_idx].uses {
            left_idx
        } else {
            right_idx
        };

        if queue[target].uses > queue[index].uses {
            break;
        }

        queue.swap_indices(index, target);
        index = target;
    }

    let mut parent_index = (index.saturating_sub(1)) / 2;
    while index > 0 && queue[index].uses < queue[parent_index].uses {
        queue.swap_indices(index, parent_index);
        index = parent_index;
        parent_index = (index.saturating_sub(1)) / 2;
    }

    index
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::Lfu;

    fn assert_heap_property<Key, Value>(lfu: &Lfu<Key, Value>) {
        for i in 0..lfu.queue.len() {
            let left_idx = i * 2 + 1;
            let right_idx = i * 2 + 2;

            if left_idx < lfu.queue.len() {
                let parent_uses = lfu.queue[i].uses;
                let left_child_uses = lfu.queue[left_idx].uses;
                assert!(
                    parent_uses <= left_child_uses,
                    "Min-heap property violated: parent at index {} has uses {}, left child at index {} has uses {}",
                    i,
                    parent_uses,
                    left_idx,
                    left_child_uses
                );
            }

            if right_idx < lfu.queue.len() {
                let parent_uses = lfu.queue[i].uses;
                let right_child_uses = lfu.queue[right_idx].uses;
                assert!(
                    parent_uses <= right_child_uses,
                    "Min-heap property violated: parent at index {} has uses {}, right child at index {} has uses {}",
                    i,
                    parent_uses,
                    right_idx,
                    right_child_uses
                );
            }
        }
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
    fn test_lfu_cache_get_updates_uses() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let least_frequent_before = cache.tail().map(|(k, _)| *k);

        cache.get(least_frequent_before.as_ref().unwrap());

        let least_frequent_after = cache.tail().map(|(k, _)| *k);
        assert_ne!(least_frequent_before, least_frequent_after);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_peek_does_not_update_uses() {
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
        assert!(cache.queue[0].value == "three");
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
    fn test_lfu_cache_equal_frequencies() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.get(&1);
        cache.get(&2);
        cache.get(&3);

        cache.insert(4, "four");

        assert_eq!(cache.len(), 3);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_mixed_frequency_patterns() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.get(&1);
        for _ in 0..2 {
            cache.get(&2);
        }
        for _ in 0..3 {
            cache.get(&3);
        }

        cache.insert(6, "value_6".to_string());
        cache.insert(7, "value_7".to_string());

        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&1));
        assert_eq!(cache.len(), 5);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_get_or_insert_with_frequency_update() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");

        let value = cache.get_or_insert_with(1, |_| "new_one");
        assert_eq!(*value, "one");

        let value = cache.get_or_insert_with(3, |_| "three");
        assert_eq!(*value, "three");

        cache.insert(4, "four");

        assert!(cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_from_iter_with_duplicates() {
        let items = vec![
            (1, "one"),
            (2, "two"),
            (3, "three"),
            (1, "one_updated"),
            (4, "four"),
            (2, "two_updated"),
        ];

        let cache: Lfu<i32, &str> = items.into_iter().collect();

        assert_eq!(cache.len(), 4);
        assert_eq!(cache.peek(&1), Some(&"one_updated"));
        assert_eq!(cache.peek(&2), Some(&"two_updated"));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_cache_insert_mut_frequency_behavior() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        let val1 = cache.insert_mut(1, String::from("one"));
        val1.push_str("_modified");

        cache.insert(2, String::from("two"));
        cache.insert(3, String::from("three"));

        let val1_again = cache.get_mut(&1).unwrap();
        val1_again.push_str("_again");

        cache.insert(4, String::from("four"));

        assert!(cache.contains_key(&1));
        assert_eq!(cache.peek(&1), Some(&String::from("one_modified_again")));
        assert_eq!(cache.len(), 3);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_empty_cache() {
        let mut cache = Lfu::<i32, i32>::new(NonZeroUsize::new(3).unwrap());

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 3);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.peek(&1), None);
        assert_eq!(cache.remove(&1), None);
        assert_eq!(cache.pop(), None);
        assert!(cache.tail().is_none());
        assert!(!cache.contains_key(&1));
    }

    #[test]
    fn test_lfu_capacity_constraints() {
        let cache = Lfu::<i32, i32>::new(NonZeroUsize::new(5).unwrap());
        assert_eq!(cache.capacity(), 5);

        let cache = Lfu::<i32, i32>::new(NonZeroUsize::new(1).unwrap());
        assert_eq!(cache.capacity(), 1);

        let cache = Lfu::<i32, i32>::new(NonZeroUsize::new(100).unwrap());
        assert_eq!(cache.capacity(), 100);
    }

    #[test]
    fn test_lfu_heap_property_maintenance() {
        let mut cache = Lfu::new(NonZeroUsize::new(10).unwrap());

        for i in 0..10 {
            cache.insert(i, format!("value_{}", i));
            assert_heap_property(&cache);
        }

        for i in 0..5 {
            cache.get(&i);
            cache.get(&i);
            assert_heap_property(&cache);
        }

        for i in 10..15 {
            cache.insert(i, format!("value_{}", i));
            assert_heap_property(&cache);
        }

        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_lfu_frequency_consistency() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");
        cache.insert(5, "five");

        cache.get(&1);
        cache.get(&1);
        cache.get(&3);
        cache.get(&2);

        cache.insert(6, "six");

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&6));

        assert!(!(cache.contains_key(&4) && cache.contains_key(&5)));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_saturating_frequency_counter() {
        let mut cache = Lfu::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");

        for _ in 0..1000 {
            cache.get(&1);
        }

        assert!(cache.queue[0].uses > 0);

        cache.insert(3, "three");

        assert!(cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_tie_breaking_behavior() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.insert(4, "four");

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&4));

        let remaining_count = [1, 2, 3]
            .iter()
            .filter(|&&k| cache.contains_key(&k))
            .count();
        assert_eq!(remaining_count, 2);

        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_complex_frequency_scenarios() {
        let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());

        cache.insert("A", 1);
        cache.insert("B", 2);
        cache.insert("C", 3);
        cache.insert("D", 4);

        for _ in 0..5 {
            cache.get(&"A");
        }
        for _ in 0..3 {
            cache.get(&"B");
        }
        for _ in 0..2 {
            cache.get(&"C");
        }
        cache.get(&"D");

        cache.insert("E", 5);
        assert!(cache.contains_key(&"A"));
        assert!(cache.contains_key(&"B"));
        assert!(cache.contains_key(&"C"));
        assert!(!cache.contains_key(&"D"));
        assert!(cache.contains_key(&"E"));

        cache.insert("F", 6);
        assert!(cache.contains_key(&"A"));
        assert!(cache.contains_key(&"B"));
        assert!(cache.contains_key(&"C"));
        assert!(!cache.contains_key(&"E"));
        assert!(cache.contains_key(&"F"));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_remove_operations() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        for i in 1..=5 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.get(&1);
        cache.get(&1);
        cache.get(&2);
        cache.get(&3);

        let removed = cache.remove(&2);
        assert_eq!(removed, Some("value_2".to_string()));
        assert!(!cache.contains_key(&2));
        assert_eq!(cache.len(), 4);
        assert_heap_property(&cache);

        let removed = cache.remove(&1);
        assert_eq!(removed, Some("value_1".to_string()));
        assert!(!cache.contains_key(&1));
        assert_eq!(cache.len(), 3);
        assert_heap_property(&cache);

        let removed = cache.remove(&10);
        assert_eq!(removed, None);
        assert_eq!(cache.len(), 3);
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_get_or_insert_with_frequency_implications() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, 10);
        cache.insert(2, 20);

        let val = cache.get_or_insert_with(1, |_| 999);
        assert_eq!(*val, 10);

        let val = cache.get_or_insert_with(3, |_| 30);
        assert_eq!(*val, 30);
        assert_eq!(cache.len(), 3);

        cache.insert(4, 40);

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert_eq!(cache.len(), 3);

        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_mutable_references() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        let val = cache.insert_mut(1, String::from("test"));
        val.push_str("_modified");

        cache.insert(2, String::from("second"));
        cache.insert(3, String::from("third"));

        if let Some(val) = cache.get_mut(&1) {
            val.push_str("_again");
        }

        cache.insert(4, String::from("fourth"));

        assert!(cache.contains_key(&1));
        assert_eq!(cache.peek(&1), Some(&String::from("test_modified_again")));
        assert!(cache.contains_key(&4));
        assert_eq!(cache.len(), 3);

        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_edge_cases() {
        let mut cache = Lfu::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one");
        assert_eq!(cache.len(), 1);
        assert_heap_property(&cache);

        cache.insert(2, "two");
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert_heap_property(&cache);

        cache.get(&2);
        cache.insert(3, "three");
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    fn test_lfu_stress_heap_maintenance() {
        let mut cache = Lfu::new(NonZeroUsize::new(20).unwrap());

        for i in 0..20 {
            cache.insert(i, i);
            assert_heap_property(&cache);
        }

        for i in 0..100 {
            match i % 4 {
                0 => {
                    cache.insert(i + 100, i);
                    assert_heap_property(&cache);
                }
                1 => {
                    cache.get(&(i % 20));
                    assert_heap_property(&cache);
                }
                2 => {
                    cache.remove(&(i % 50));
                    assert_heap_property(&cache);
                }
                3 => {
                    cache.get_or_insert_with(i + 200, |k| *k);
                    assert_heap_property(&cache);
                }
                _ => unreachable!(),
            }
        }

        assert!(cache.len() <= 20);
    }

    #[test]
    fn test_lfu_frequency_after_removal() {
        let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");

        cache.get(&1);
        cache.get(&2);
        cache.get(&2);
        cache.get(&3);
        cache.get(&3);
        cache.get(&3);

        cache.remove(&3);
        assert_heap_property(&cache);

        cache.insert(5, "five");
        cache.get(&5);

        cache.insert(6, "six");

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(!cache.contains_key(&4));
        assert!(cache.contains_key(&5));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_collection_traits() {
        let items = vec![(1, "one"), (2, "two"), (3, "three")];

        let cache: Lfu<i32, &str> = items.into_iter().collect();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert_heap_property(&cache);

        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());
        cache.insert(1, 10);

        let more_items = vec![(2, 20), (3, 30)];
        cache.extend(more_items);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_clear_and_shrink() {
        let mut cache = Lfu::new(NonZeroUsize::new(10).unwrap());

        for i in 0..5 {
            cache.insert(i, i * 10);
        }

        assert_eq!(cache.len(), 5);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 10);
        assert!(cache.tail().is_none());

        cache.shrink_to_fit();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lfu_boundary_frequency_values() {
        let mut cache = Lfu::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "low_freq");
        cache.insert(2, "high_freq");

        for _ in 0..1000 {
            cache.get(&2);
        }

        let high_freq_uses = cache
            .queue
            .iter()
            .find(|(_, entry)| entry.value == "high_freq")
            .map(|(_, entry)| entry.uses)
            .unwrap();

        assert!(high_freq_uses > 1000);
        assert!(high_freq_uses < u64::MAX);

        cache.insert(3, "new");

        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_lfu_heap_structure_after_operations() {
        let mut cache = Lfu::new(NonZeroUsize::new(7).unwrap());

        for i in 1..=7 {
            cache.insert(i, format!("value_{}", i));
        }

        for i in 1..=7 {
            for _ in 0..i {
                cache.get(&i);
            }
        }

        assert_heap_property(&cache);

        cache.remove(&1);
        assert_heap_property(&cache);

        cache.remove(&7);
        assert_heap_property(&cache);

        cache.remove(&4);
        assert_heap_property(&cache);

        cache.insert(8, "value_8".to_string());
        assert_heap_property(&cache);

        cache.insert(9, "value_9".to_string());
        assert_heap_property(&cache);

        assert_eq!(cache.len(), 6);
    }

    #[test]
    fn test_lfu_consistent_eviction_policy() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert("low", 1);
        cache.insert("medium", 2);
        cache.insert("high", 3);

        cache.get(&"medium");
        cache.get(&"high");
        cache.get(&"high");

        cache.insert("new1", 4);
        assert!(!cache.contains_key(&"low"));
        assert!(cache.contains_key(&"medium"));
        assert!(cache.contains_key(&"high"));
        assert!(cache.contains_key(&"new1"));

        cache.insert("new2", 5);
        assert!(cache.contains_key(&"medium"));
        assert!(cache.contains_key(&"high"));
        assert!(!cache.contains_key(&"new1"));
        assert!(cache.contains_key(&"new2"));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_from_iterator_overlapping_keys_frequency_tracking() {
        let items = vec![
            (1, "first"),
            (2, "second"),
            (1, "updated_first"),
            (3, "third"),
            (2, "updated_second"),
            (1, "final_first"),
        ];

        let mut cache: Lfu<i32, &str> = items.into_iter().collect();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);

        assert_eq!(cache.peek(&1), Some(&"final_first"));
        assert_eq!(cache.peek(&2), Some(&"updated_second"));
        assert_eq!(cache.peek(&3), Some(&"third"));

        assert_eq!(cache.tail(), Some((&3, &"third")));

        cache.insert(4, "four");

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(cache.contains_key(&4));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_from_iterator_many_overlapping_keys() {
        let items = vec![
            (1, 100),
            (1, 200),
            (1, 300),
            (2, 400),
            (1, 500),
            (2, 600),
            (1, 700),
            (3, 800),
        ];

        let mut cache: Lfu<i32, i32> = items.into_iter().collect();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);

        assert_eq!(cache.peek(&1), Some(&700));
        assert_eq!(cache.peek(&2), Some(&600));
        assert_eq!(cache.peek(&3), Some(&800));

        assert_eq!(cache.tail(), Some((&3, &800)));

        cache.insert(4, 900);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(cache.contains_key(&4));

        cache.insert(5, 1000);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&4));
        assert!(cache.contains_key(&5));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_from_iterator_frequency_vs_insertion_order() {
        let items = vec![
            (1, "one"),
            (2, "two"),
            (3, "three"),
            (2, "updated_two"),
            (4, "four"),
            (3, "updated_three"),
            (3, "final_three"),
        ];

        let mut cache: Lfu<i32, &str> = items.into_iter().collect();

        assert_eq!(cache.len(), 4);

        cache.insert(5, "five");

        assert!(cache.contains_key(&3));
        assert_eq!(cache.peek(&3), Some(&"final_three"));

        assert!(cache.contains_key(&2));

        assert!(cache.contains_key(&5));

        assert!(!(cache.contains_key(&1) && cache.contains_key(&4)));

        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_from_iterator_empty_and_single_item() {
        let empty_items: Vec<(i32, &str)> = vec![];
        let cache: Lfu<i32, &str> = empty_items.into_iter().collect();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 1);

        let single_item = vec![(1, "one")];
        let cache: Lfu<i32, &str> = single_item.into_iter().collect();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.capacity(), 1);
        assert_eq!(cache.peek(&1), Some(&"one"));
        assert_heap_property(&cache);

        let overlapping_single = vec![(1, "first"), (1, "second"), (1, "third")];
        let cache: Lfu<i32, &str> = overlapping_single.into_iter().collect();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.capacity(), 1);
        assert_eq!(cache.peek(&1), Some(&"third"));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_basic() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");
        cache.insert(5, "five");

        cache.retain(|&k, _| k % 2 == 0);

        assert_eq!(cache.len(), 2);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert!(!cache.contains_key(&5));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_with_frequency_considerations() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");
        cache.insert(5, "five");

        cache.get(&1);
        cache.get(&2);
        cache.get(&2);
        cache.get(&3);
        cache.get(&3);
        cache.get(&3);

        cache.retain(|&k, v| k <= 3 && v.len() > 3);

        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(!cache.contains_key(&4));
        assert!(!cache.contains_key(&5));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_all() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.retain(|_, _| true);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_none() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.retain(|_, _| false);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(!cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
    }

    #[test]
    fn test_heap_lfu_retain_empty_cache() {
        let mut cache = Lfu::<i32, &str>::new(NonZeroUsize::new(3).unwrap());

        cache.retain(|_, _| true);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_heap_lfu_retain_single_item() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");

        cache.retain(|&k, _| k == 1);

        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&1));
        assert_heap_property(&cache);

        cache.retain(|&k, _| k != 1);

        assert_eq!(cache.len(), 0);
        assert!(!cache.contains_key(&1));
    }

    #[test]
    fn test_heap_lfu_retain_with_heap_property_maintenance() {
        let mut cache = Lfu::new(NonZeroUsize::new(10).unwrap());

        for i in 1..=10 {
            cache.insert(i, format!("value_{}", i));

            for _ in 0..i {
                cache.get(&i);
            }
        }

        assert_heap_property(&cache);

        cache.retain(|&k, _| k % 2 == 1);

        assert_eq!(cache.len(), 5);
        for i in 1..=10 {
            if i % 2 == 1 {
                assert!(cache.contains_key(&i));
            } else {
                assert!(!cache.contains_key(&i));
            }
        }
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_frequency_preservation() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.get(&2);
        cache.get(&2);
        cache.get(&3);
        cache.get(&3);
        cache.get(&3);

        cache.retain(|&k, _| k == 1 || k == 3);

        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));

        cache.insert(4, "four");
        cache.get(&4);
        cache.insert(5, "five");
        cache.get(&5);
        cache.insert(6, "six");
        cache.get(&6);
        cache.insert(7, "seven");

        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert!(cache.contains_key(&5));
        assert!(cache.contains_key(&6));
        assert!(cache.contains_key(&7));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_with_mutable_values() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, String::from("short"));
        cache.insert(2, String::from("medium"));
        cache.insert(3, String::from("very_long_string"));
        cache.insert(4, String::from("tiny"));
        cache.insert(5, String::from("another_long_string"));

        cache.retain(|_, v| v.len() > 5);

        assert_eq!(cache.len(), 3);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(!cache.contains_key(&4));
        assert!(cache.contains_key(&5));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_complex_predicate() {
        let mut cache = Lfu::new(NonZeroUsize::new(6).unwrap());

        cache.insert(1, 10);
        cache.insert(2, 25);
        cache.insert(3, 30);
        cache.insert(4, 45);
        cache.insert(5, 50);
        cache.insert(6, 65);

        cache.get(&1);
        cache.get(&1);
        cache.get(&3);
        cache.get(&5);
        cache.get(&5);
        cache.get(&5);

        cache.retain(|&k, &mut v| k % 2 == 1 && v % 5 == 0);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(!cache.contains_key(&4));
        assert!(cache.contains_key(&5));
        assert!(!cache.contains_key(&6));
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_stress_test() {
        let mut cache = Lfu::new(NonZeroUsize::new(100).unwrap());

        for i in 0..100 {
            cache.insert(i, i * 10);

            for _ in 0..(i % 5) {
                cache.get(&i);
            }
        }

        assert_eq!(cache.len(), 100);
        assert_heap_property(&cache);

        cache.retain(|&k, _| k % 3 == 0);

        let expected_len = (0..100).filter(|&x| x % 3 == 0).count();
        assert_eq!(cache.len(), expected_len);

        for i in 0..100 {
            if i % 3 == 0 {
                assert!(cache.contains_key(&i));
            } else {
                assert!(!cache.contains_key(&i));
            }
        }
        assert_heap_property(&cache);
    }

    #[test]
    fn test_heap_lfu_retain_after_operations() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");
        cache.insert(5, "five");

        cache.get(&1);
        cache.get(&3);
        cache.remove(&2);
        cache.get(&4);

        assert_eq!(cache.len(), 4);

        cache.retain(|&k, _| k > 2);

        assert_eq!(cache.len(), 3);
        assert!(!cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
        assert!(cache.contains_key(&5));
        assert_heap_property(&cache);

        cache.insert(6, "six");
        cache.get(&3);
        assert_eq!(cache.len(), 4);
        assert_heap_property(&cache);
    }
}
