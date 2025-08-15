//! List-based Least Frequently Used (LFU) cache implementation with O(1)
//! operations.
//!
//! This module provides an O(1) LFU eviction policy using frequency buckets
//! and doubly-linked lists to track the least frequently used items.

use std::{
    collections::BTreeMap,
    hash::Hash,
};

use indexmap::IndexMap;

use crate::{
    EntryValue,
    Metadata,
    Policy,
    RandomState,
    private,
};

/// List-based LFU cache eviction policy implementation with O(1) operations.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct LfuPolicy;

impl private::Sealed for LfuPolicy {}

/// List-based LFU cache entry with frequency tracking and list pointers.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct LfuEntry<T> {
    value: T,
    frequency: u64,
    prev: Option<usize>,
    next: Option<usize>,
}

impl<T> private::Sealed for LfuEntry<T> {}

impl<T> EntryValue<T> for LfuEntry<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            frequency: 0,
            prev: None,
            next: None,
        }
    }

    fn prepare_for_reinsert(&mut self) {
        self.prev = None;
        self.next = None;
        self.frequency = self.frequency.saturating_sub(1)
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

/// Metadata for list-based LFU policy tracking frequency buckets.
#[derive(Debug, Clone, Default)]
#[doc(hidden)]
pub struct LfuMetadata {
    frequency_head_tail: BTreeMap<u64, (usize, usize)>,
}

impl private::Sealed for LfuMetadata {}

impl Metadata for LfuMetadata {
    fn tail_index(&self) -> usize {
        self.frequency_head_tail
            .iter()
            .next()
            .map_or(0, |(_, &(_, tail))| tail)
    }
}

impl<T> Policy<T> for LfuPolicy {
    type EntryType = LfuEntry<T>;
    type MetadataType = LfuMetadata;

    fn touch_entry(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<impl Hash + Eq, Self::EntryType, RandomState>,
    ) -> usize {
        if index >= queue.len() {
            return index;
        }

        unlink_node(index, metadata, queue);

        queue[index].frequency = queue[index].frequency.saturating_add(1);

        link_node(index, metadata, queue);

        index
    }

    fn swap_remove_entry<K: Hash + Eq>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<K, Self::EntryType, RandomState>,
    ) -> (usize, Option<(K, Self::EntryType)>) {
        if index >= queue.len() {
            return (index, None);
        }

        unlink_node(index, metadata, queue);

        let result = queue.swap_remove_index(index);
        if index == queue.len() {
            return (index, result);
        }

        if let Some(prev) = queue[index].prev {
            queue[prev].next = Some(index);
        }
        if let Some(next) = queue[index].next {
            queue[next].prev = Some(index);
        }

        if let Some((head_idx, tail_idx)) = metadata
            .frequency_head_tail
            .get_mut(&queue[index].frequency)
        {
            if *head_idx == queue.len() {
                *head_idx = index;
            }
            if *tail_idx == queue.len() {
                *tail_idx = index;
            }
        }

        (index, result)
    }
}

fn unlink_node<T>(
    index: usize,
    metadata: &mut LfuMetadata,
    queue: &mut IndexMap<impl Hash + Eq, LfuEntry<T>, RandomState>,
) {
    let frequency = queue[index].frequency;
    let prev = queue[index].prev;
    let next = queue[index].next;

    if let Some(prev_idx) = prev {
        queue[prev_idx].next = next;
    }
    if let Some(next_idx) = next {
        queue[next_idx].prev = prev;
    }

    if let Some((head_idx, tail_idx)) = metadata.frequency_head_tail.get_mut(&frequency) {
        if *head_idx == index {
            if let Some(next_idx) = next {
                *head_idx = next_idx;
            } else {
                metadata.frequency_head_tail.remove(&frequency);
            }
        } else if *tail_idx == index {
            if let Some(prev_idx) = prev {
                *tail_idx = prev_idx;
            } else {
                metadata.frequency_head_tail.remove(&frequency);
            }
        }
    }

    queue[index].prev = None;
    queue[index].next = None;
}

fn link_node<T>(
    index: usize,
    metadata: &mut LfuMetadata,
    queue: &mut IndexMap<impl Hash + Eq, LfuEntry<T>, RandomState>,
) {
    metadata
        .frequency_head_tail
        .entry(queue[index].frequency)
        .and_modify(|(head_idx, _)| {
            queue[*head_idx].prev = Some(index);
            queue[index].next = Some(*head_idx);
            *head_idx = index;
        })
        .or_insert_with(|| {
            queue[index].prev = None;
            queue[index].next = None;
            (index, index)
        });
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::Lfu;

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
    fn test_lfu_cache_get_updates_uses() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let least_frequent_before = cache.tail().map(|(k, _)| *k);

        cache.get(least_frequent_before.as_ref().unwrap());

        let least_frequent_after = cache.tail().map(|(k, _)| *k);
        assert_ne!(least_frequent_before, least_frequent_after);
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
        dbg!(&cache);

        assert!(cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
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
        }

        for i in 0..5 {
            cache.get(&i);
            cache.get(&i);
        }

        for i in 10..15 {
            cache.insert(i, format!("value_{}", i));
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

        let removed = cache.remove(&1);
        assert_eq!(removed, Some("value_1".to_string()));
        assert!(!cache.contains_key(&1));
        assert_eq!(cache.len(), 3);

        let removed = cache.remove(&10);
        assert_eq!(removed, None);
        assert_eq!(cache.len(), 3);
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
    }

    #[test]
    fn test_lfu_edge_cases() {
        let mut cache = Lfu::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one");
        assert_eq!(cache.len(), 1);

        cache.insert(2, "two");
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));

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
        }

        for i in 0..100 {
            match i % 4 {
                0 => {
                    cache.insert(i + 100, i);
                }
                1 => {
                    cache.get(&(i % 20));
                }
                2 => {
                    cache.remove(&(i % 50));
                }
                3 => {
                    cache.get_or_insert_with(i + 200, |k| *k);
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

        cache.insert(5, "five");
        cache.get(&5);

        cache.insert(6, "six");

        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(!cache.contains_key(&3));
        assert!(!cache.contains_key(&4));
        assert!(cache.contains_key(&5));
    }

    #[test]
    fn test_lfu_collection_traits() {
        let items = vec![(1, "one"), (2, "two"), (3, "three")];

        let cache: Lfu<i32, &str> = items.into_iter().collect();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);

        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());
        cache.insert(1, 10);

        let more_items = vec![(2, 20), (3, 30)];
        cache.extend(more_items);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
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
    }

    #[test]
    fn test_lfu_from_iterator_overlapping_keys_frequency_tracking() {
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
    }

    #[test]
    fn test_lfu_from_iterator_many_overlapping_keys() {
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
    }

    #[test]
    fn test_lfu_from_iterator_frequency_vs_insertion_order() {
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
    }

    #[test]
    fn test_lfu_from_iterator_empty_and_single_item() {
        let empty_items: Vec<(i32, &str)> = vec![];
        let cache: Lfu<i32, &str> = empty_items.into_iter().collect();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 1);

        let single_item = vec![(1, "one")];
        let cache: Lfu<i32, &str> = single_item.into_iter().collect();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.capacity(), 1);
        assert_eq!(cache.peek(&1), Some(&"one"));

        let overlapping_single = vec![(1, "first"), (1, "second"), (1, "third")];
        let cache: Lfu<i32, &str> = overlapping_single.into_iter().collect();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.capacity(), 1);
        assert_eq!(cache.peek(&1), Some(&"third"));
    }

    #[test]
    fn test_lfu_retain_basic() {
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
    }

    #[test]
    fn test_lfu_retain_with_frequency_considerations() {
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
    }

    #[test]
    fn test_lfu_retain_all() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.retain(|_, _| true);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    fn test_lfu_retain_none() {
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
    fn test_lfu_retain_empty_cache() {
        let mut cache = Lfu::<i32, &str>::new(NonZeroUsize::new(3).unwrap());

        cache.retain(|_, _| true);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lfu_retain_single_item() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");

        cache.retain(|&k, _| k == 1);

        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&1));

        cache.retain(|&k, _| k != 1);

        assert_eq!(cache.len(), 0);
        assert!(!cache.contains_key(&1));
    }

    #[test]
    fn test_lfu_retain_with_heap_property_maintenance() {
        let mut cache = Lfu::new(NonZeroUsize::new(10).unwrap());

        for i in 1..=10 {
            cache.insert(i, format!("value_{}", i));

            for _ in 0..i {
                cache.get(&i);
            }
        }

        cache.retain(|&k, _| k % 2 == 1);

        assert_eq!(cache.len(), 5);
        for i in 1..=10 {
            if i % 2 == 1 {
                assert!(cache.contains_key(&i));
            } else {
                assert!(!cache.contains_key(&i));
            }
        }
    }

    #[test]
    fn test_lfu_retain_frequency_preservation() {
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
    }

    #[test]
    fn test_lfu_retain_with_mutable_values() {
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
    }

    #[test]
    fn test_lfu_retain_complex_predicate() {
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
    }

    #[test]
    fn test_lfu_retain_stress_test() {
        let mut cache = Lfu::new(NonZeroUsize::new(100).unwrap());

        for i in 0..100 {
            cache.insert(i, i * 10);

            for _ in 0..(i % 5) {
                cache.get(&i);
            }
        }

        assert_eq!(cache.len(), 100);

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
    }

    #[test]
    fn test_lfu_retain_after_operations() {
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

        cache.insert(6, "six");
        cache.get(&3);
        assert_eq!(cache.len(), 4);
    }
}
