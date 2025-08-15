//! Least Recently Used (LRU) cache implementation.
//!
//! This module provides the LRU eviction policy, which removes the least
//! recently accessed entry when the cache reaches capacity.

use std::hash::Hash;

use indexmap::IndexMap;

use crate::{
    EntryValue,
    Metadata,
    Policy,
    RandomState,
    private,
};

/// LRU cache eviction policy implementation.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct LruPolicy;

impl private::Sealed for LruPolicy {}

/// LRU cache entry containing value and linked list pointers.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct LruEntry<T> {
    value: T,
    next: Option<usize>,
    prev: Option<usize>,
}

impl<T> private::Sealed for LruEntry<T> {}

impl<T> EntryValue<T> for LruEntry<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            next: None,
            prev: None,
        }
    }

    fn prepare_for_reinsert(&mut self) {
        self.next = None;
        self.prev = None;
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

/// Metadata for LRU policy tracking head, tail, and aging counter.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
#[derive(Default)]
pub struct LruMetadata {
    head: usize,
    tail: usize,
}

impl private::Sealed for LruMetadata {}

impl Metadata for LruMetadata {
    fn tail_index(&self) -> usize {
        self.tail
    }
}

impl<T> Policy<T> for LruPolicy {
    type EntryType = LruEntry<T>;
    type MetadataType = LruMetadata;

    fn touch_entry(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<impl Hash + Eq, Self::EntryType, RandomState>,
    ) -> usize {
        if index >= queue.len() {
            return index;
        }

        // Move the accessed entry to the head of the queue
        let old_head = metadata.head;
        if old_head == index {
            return index;
        }

        metadata.head = index;
        let old_prev = queue[index].prev;
        let old_next = queue[index].next;
        queue[index].next = None;
        queue[index].prev = Some(old_head);

        if metadata.tail == index {
            metadata.tail = old_next.unwrap_or_default();
        }

        queue[old_head].next = Some(index);

        if let Some(prev) = old_prev {
            queue[prev].next = old_next;
        }

        if let Some(next) = old_next {
            queue[next].prev = old_prev;
        }
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

        if queue.len() == 1 {
            // If there's only one entry, just remove it
            return (index, queue.swap_remove_index(index));
        }

        let (k, e) = queue.swap_remove_index(index).unwrap();
        if queue.len() == 1 {
            metadata.head = 0;
            metadata.tail = 0;
            queue[0].prev = None;
            queue[0].next = None;
            return (index, Some((k, e)));
        }

        if index == metadata.head {
            // Head was removed, update it
            metadata.head = e.prev.unwrap_or_default();
        }
        if metadata.head == queue.len() {
            // Head was pointing to the perturbed index
            metadata.head = index;
        }

        if index == metadata.tail {
            metadata.tail = e.next.unwrap_or_default();
        }
        if metadata.tail == queue.len() {
            // Tail was pointing to the perturbed index
            metadata.tail = index;
        }

        // Unlink the entry from the queue
        if let Some(prev) = e.prev {
            if prev == queue.len() {
                queue[index].next = None;
            } else {
                queue[prev].next = e.next;
            }
        }
        if let Some(next) = e.next {
            if next == queue.len() {
                queue[index].prev = None;
            } else {
                queue[next].prev = e.prev;
            }
        }

        if index == queue.len() {
            return (index, Some((k, e)));
        }

        // Update the links for the moved entry
        if let Some(next) = queue[index].next {
            debug_assert_eq!(queue[next].prev, Some(queue.len()));
            queue[next].prev = Some(index);
        }

        if let Some(prev) = queue[index].prev {
            debug_assert_eq!(queue[prev].next, Some(queue.len()));
            queue[prev].next = Some(index);
        }

        (index, Some((k, e)))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::Lru;

    #[test]
    fn test_lru_trivial() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());
        lru.insert("a", 1);
        lru.insert("b", 2);
        lru.insert("c", 3);

        assert_eq!(lru.get(&"a"), Some(&1));
        assert_eq!(lru.get(&"b"), Some(&2));
        assert_eq!(lru.get(&"c"), Some(&3));

        lru.get(&"a");
        lru.insert("d", 4);

        assert_eq!(lru.get(&"a"), Some(&1));
        assert_eq!(lru.get(&"b"), None);
    }

    #[test]
    fn test_lru_trivial_get_or_insert() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());
        lru.insert("a", 1);
        lru.insert("b", 2);
        lru.insert("c", 3);

        assert_eq!(lru.get_or_insert_with("a", |_| 10), &1);
        assert_eq!(lru.get_or_insert_with("d", |_| 4), &4);
        assert_eq!(lru.get(&"a"), Some(&1));
        assert_eq!(lru.get(&"b"), None);
    }

    #[test]
    fn test_lru_eviction_order() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        lru.insert(4, 40);

        assert_eq!(lru.get(&1), None);
        assert_eq!(lru.get(&2), Some(&20));
        assert_eq!(lru.get(&3), Some(&30));
        assert_eq!(lru.get(&4), Some(&40));
    }

    #[test]
    fn test_lru_access_updates_order() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        lru.get(&1);

        lru.insert(4, 40);

        assert_eq!(lru.get(&1), Some(&10));
        assert_eq!(lru.get(&2), None);
        assert_eq!(lru.get(&3), Some(&30));
        assert_eq!(lru.get(&4), Some(&40));
    }

    #[test]
    fn test_lru_multiple_accesses() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);

        lru.get(&1);
        lru.get(&1);
        lru.get(&1);

        lru.insert(3, 30);

        assert_eq!(lru.get(&1), Some(&10));
        assert_eq!(lru.get(&2), None);
        assert_eq!(lru.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_update_existing_key() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);

        lru.insert(1, 100);

        lru.insert(3, 30);

        assert_eq!(lru.get(&1), Some(&100));
        assert_eq!(lru.get(&2), None);
        assert_eq!(lru.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_single_capacity() {
        let mut lru = Lru::new(NonZeroUsize::new(1).unwrap());

        lru.insert(1, 10);
        assert_eq!(lru.get(&1), Some(&10));

        lru.insert(2, 20);
        assert_eq!(lru.get(&1), None);
        assert_eq!(lru.get(&2), Some(&20));

        lru.get(&2);
        lru.insert(3, 30);
        assert_eq!(lru.get(&2), None);
        assert_eq!(lru.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_complex_access_pattern() {
        let mut lru = Lru::new(NonZeroUsize::new(4).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);
        lru.insert(4, 40);

        lru.get(&2);
        lru.get(&1);
        lru.get(&3);

        lru.insert(5, 50);

        assert_eq!(lru.get(&1), Some(&10));
        assert_eq!(lru.get(&2), Some(&20));
        assert_eq!(lru.get(&3), Some(&30));
        assert_eq!(lru.get(&4), None);
        assert_eq!(lru.get(&5), Some(&50));
    }

    #[test]
    fn test_lru_interleaved_operations() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.get(&1);
        lru.insert(3, 30);
        lru.get(&2);
        lru.insert(4, 40);

        assert_eq!(lru.get(&1), None);
        assert_eq!(lru.get(&2), Some(&20));
        assert_eq!(lru.get(&3), Some(&30));
        assert_eq!(lru.get(&4), Some(&40));
    }

    #[test]
    fn test_lru_get_or_insert_with_eviction() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);

        lru.get(&1);

        let value = lru.get_or_insert_with(3, |_| 30);
        assert_eq!(value, &30);

        assert_eq!(lru.get(&1), Some(&10));
        assert_eq!(lru.get(&2), None);
        assert_eq!(lru.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_get_or_insert_existing_key() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        let value = lru.get_or_insert_with(1, |_| 999);
        assert_eq!(value, &10);

        lru.insert(4, 40);

        assert_eq!(lru.get(&1), Some(&10));
        assert_eq!(lru.get(&2), None);
        assert_eq!(lru.get(&3), Some(&30));
        assert_eq!(lru.get(&4), Some(&40));
    }

    #[test]
    fn test_lru_sequential_pattern() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        for i in 1..=10 {
            lru.insert(i, i * 10);
        }

        assert_eq!(lru.get(&8), Some(&80));
        assert_eq!(lru.get(&9), Some(&90));
        assert_eq!(lru.get(&10), Some(&100));
        assert_eq!(lru.get(&7), None);
        assert_eq!(lru.get(&1), None);
    }

    #[test]
    fn test_lru_repeated_access_single_key() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);

        for _ in 0..100 {
            lru.get(&1);
        }

        lru.insert(3, 30);

        assert_eq!(lru.get(&1), Some(&10));
        assert_eq!(lru.get(&2), None);
        assert_eq!(lru.get(&3), Some(&30));
    }

    #[test]
    fn test_lru_empty_cache() {
        let mut lru = Lru::<i32, i32>::new(NonZeroUsize::new(3).unwrap());

        assert!(lru.is_empty());
        assert_eq!(lru.len(), 0);
        assert_eq!(lru.capacity(), 3);
        assert_eq!(lru.get(&1), None);
        assert_eq!(lru.peek(&1), None);
        assert_eq!(lru.remove(&1), None);
        assert_eq!(lru.pop(), None);
        assert!(lru.tail().is_none());
        assert!(!lru.contains_key(&1));
    }

    #[test]
    fn test_lru_capacity_constraints() {
        let lru = Lru::<i32, i32>::new(NonZeroUsize::new(5).unwrap());
        assert_eq!(lru.capacity(), 5);

        let lru = Lru::<i32, i32>::new(NonZeroUsize::new(1).unwrap());
        assert_eq!(lru.capacity(), 1);

        let lru = Lru::<i32, i32>::new(NonZeroUsize::new(100).unwrap());
        assert_eq!(lru.capacity(), 100);
    }

    #[test]
    fn test_lru_peek_no_side_effects() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());
        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        assert_eq!(lru.peek(&1), Some(&10));

        lru.insert(4, 40);

        assert_eq!(lru.get(&1), None);
        assert_eq!(lru.get(&2), Some(&20));
        assert_eq!(lru.get(&3), Some(&30));
        assert_eq!(lru.get(&4), Some(&40));
    }

    #[test]
    fn test_lru_contains_key() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        assert!(!lru.contains_key(&1));

        lru.insert(1, 10);
        assert!(lru.contains_key(&1));
        assert!(!lru.contains_key(&2));

        lru.insert(2, 20);
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));

        lru.insert(3, 30);
        assert!(!lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&3));
    }

    #[test]
    fn test_lru_remove() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());
        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        assert_eq!(lru.remove(&2), Some(20));
        assert_eq!(lru.len(), 2);
        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&3));

        assert_eq!(lru.remove(&2), None);
        assert_eq!(lru.len(), 2);

        lru.insert(4, 40);
        assert_eq!(lru.len(), 3);
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&3));
        assert!(lru.contains_key(&4));
    }

    #[test]
    fn test_lru_pop() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());
        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        let popped = lru.pop();
        assert_eq!(popped, Some((1, 10)));
        assert_eq!(lru.len(), 2);
        assert!(!lru.contains_key(&1));

        let popped = lru.pop();
        assert_eq!(popped, Some((2, 20)));
        assert_eq!(lru.len(), 1);

        let popped = lru.pop();
        assert_eq!(popped, Some((3, 30)));
        assert_eq!(lru.len(), 0);
        assert!(lru.is_empty());

        let popped = lru.pop();
        assert_eq!(popped, None);
    }

    #[test]
    fn test_lru_tail() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        assert!(lru.tail().is_none());

        lru.insert(1, 10);
        assert_eq!(lru.tail(), Some((&1, &10)));

        lru.insert(2, 20);
        assert_eq!(lru.tail(), Some((&1, &10)));

        lru.insert(3, 30);
        assert_eq!(lru.tail(), Some((&1, &10)));

        lru.get(&1);
        assert_eq!(lru.tail(), Some((&2, &20)));
    }

    #[test]
    fn test_lru_clear() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());
        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        assert_eq!(lru.len(), 3);
        assert!(!lru.is_empty());

        lru.clear();

        assert_eq!(lru.len(), 0);
        assert!(lru.is_empty());
        assert_eq!(lru.capacity(), 3);
        assert!(!lru.contains_key(&1));
        assert!(!lru.contains_key(&2));
        assert!(!lru.contains_key(&3));
        assert!(lru.tail().is_none());
    }

    #[test]
    fn test_lru_mutable_access() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());
        lru.insert(1, String::from("hello"));
        lru.insert(2, String::from("world"));

        if let Some(val) = lru.get_mut(&1) {
            val.push_str(" modified");
        }

        assert_eq!(lru.get(&1), Some(&String::from("hello modified")));

        lru.insert(3, String::from("new"));

        assert!(lru.contains_key(&1));
        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&3));
    }

    #[test]
    fn test_lru_insert_mut() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        let val = lru.insert_mut(1, String::from("test"));
        val.push_str(" modified");

        assert_eq!(lru.get(&1), Some(&String::from("test modified")));

        let val = lru.insert_mut(1, String::from("replaced"));
        val.push_str(" again");

        assert_eq!(lru.get(&1), Some(&String::from("replaced again")));
    }

    #[test]
    fn test_lru_get_or_insert_with_mut() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());
        lru.insert(1, String::from("existing"));

        let val = lru.get_or_insert_with_mut(1, |_| String::from("new"));
        val.push_str(" modified");

        assert_eq!(lru.get(&1), Some(&String::from("existing modified")));

        let val = lru.get_or_insert_with_mut(2, |_| String::from("created"));
        val.push_str(" also modified");

        assert_eq!(lru.get(&2), Some(&String::from("created also modified")));
    }

    #[test]
    fn test_lru_extend() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());
        lru.insert(1, 10);

        let items = vec![(2, 20), (3, 30), (4, 40)];
        lru.extend(items);

        assert_eq!(lru.len(), 4);
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&3));
        assert!(lru.contains_key(&4));
    }

    #[test]
    fn test_lru_from_iterator() {
        let items = vec![(1, 10), (2, 20), (3, 30)];
        let lru: Lru<i32, i32> = items.into_iter().collect();

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.capacity(), 3);
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&3));
    }

    #[test]
    fn test_lru_from_iterator_with_eviction() {
        let items = vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)];
        let lru: Lru<i32, i32> = items.into_iter().collect();

        assert_eq!(lru.len(), 5);
        assert_eq!(lru.capacity(), 5);

        for i in 1..=5 {
            assert!(lru.contains_key(&i));
        }
    }

    #[test]
    fn test_lru_shrink_to_fit() {
        let mut lru = Lru::new(NonZeroUsize::new(10).unwrap());
        lru.insert(1, 10);
        lru.insert(2, 20);

        lru.shrink_to_fit();

        assert_eq!(lru.len(), 2);
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));
    }

    #[test]
    fn test_lru_entry_lifetime_and_ordering() {
        let mut lru = Lru::new(NonZeroUsize::new(4).unwrap());

        lru.insert(1, "first");
        lru.insert(2, "second");
        lru.insert(3, "third");
        lru.insert(4, "fourth");

        lru.get(&3);
        lru.get(&1);
        lru.get(&2);

        lru.insert(5, "fifth");

        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&3));
        assert!(!lru.contains_key(&4));
        assert!(lru.contains_key(&5));

        lru.insert(6, "sixth");

        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(!lru.contains_key(&3));
        assert!(lru.contains_key(&5));
        assert!(lru.contains_key(&6));
    }

    #[test]
    fn test_lru_stress_random_operations() {
        let mut lru = Lru::new(NonZeroUsize::new(10).unwrap());

        for i in 0..20 {
            lru.insert(i, i * 10);
        }

        assert_eq!(lru.len(), 10);

        for i in 10..20 {
            lru.get(&i);
            if i % 2 == 0 {
                lru.remove(&(i - 10));
            }
        }

        assert!(lru.len() <= 10);
        for i in 15..20 {
            assert!(lru.contains_key(&i));
        }
    }

    #[test]
    fn test_lru_boundary_conditions() {
        let mut lru = Lru::new(NonZeroUsize::new(1000).unwrap());
        for i in 0..500 {
            lru.insert(i, i);
        }
        assert_eq!(lru.len(), 500);

        for i in 500..1000 {
            lru.insert(i, i);
        }
        assert_eq!(lru.len(), 1000);

        lru.insert(1000, 1000);
        assert_eq!(lru.len(), 1000);
        assert!(!lru.contains_key(&0));
        assert!(lru.contains_key(&1000));
    }

    #[test]
    fn test_lru_consistent_state_after_operations() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 100);
        lru.insert(2, 200);
        assert_eq!(lru.len(), 2);

        lru.get(&1);
        lru.insert(3, 300);
        assert_eq!(lru.len(), 3);

        lru.remove(&2);
        assert_eq!(lru.len(), 2);
        assert!(!lru.contains_key(&2));

        lru.insert(4, 400);
        lru.insert(5, 500);
        assert_eq!(lru.len(), 3);

        assert!(lru.get(&4).is_some() || lru.get(&5).is_some());
        assert!(lru.tail().is_some());
    }

    #[test]
    fn test_lru_key_value_types() {
        let mut lru_str = Lru::new(NonZeroUsize::new(2).unwrap());
        lru_str.insert("key1", 1);
        lru_str.insert("key2", 2);
        assert!(lru_str.contains_key(&"key1"));

        let mut lru_vec = Lru::new(NonZeroUsize::new(2).unwrap());
        lru_vec.insert(1, vec![1, 2, 3]);
        lru_vec.insert(2, vec![4, 5, 6]);
        assert_eq!(lru_vec.get(&1), Some(&vec![1, 2, 3]));
    }

    #[test]
    fn test_lru_from_iterator_overlapping_keys() {
        let items = vec![
            (1, "first"),
            (2, "second"),
            (1, "updated_first"),
            (3, "third"),
            (2, "updated_second"),
            (1, "final_first"),
        ];

        let lru: Lru<i32, &str> = items.into_iter().collect();

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.capacity(), 3);

        assert_eq!(lru.peek(&1), Some(&"final_first"));
        assert_eq!(lru.peek(&2), Some(&"updated_second"));
        assert_eq!(lru.peek(&3), Some(&"third"));

        assert_eq!(lru.tail(), Some((&3, &"third")));
    }

    #[test]
    fn test_lru_from_iterator_overlapping_with_eviction() {
        let items = vec![
            (1, "one"),
            (2, "two"),
            (3, "three"),
            (4, "four"),
            (5, "five"),
            (1, "updated_one"),
            (6, "six"),
        ];

        let lru: Lru<i32, &str> = items.into_iter().collect();

        assert_eq!(lru.len(), 6);
        assert_eq!(lru.capacity(), 6);

        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&3));
        assert!(lru.contains_key(&4));
        assert!(lru.contains_key(&5));
        assert!(lru.contains_key(&6));

        assert_eq!(lru.peek(&1), Some(&"updated_one"));
    }

    #[test]
    fn test_lru_from_iterator_many_duplicates() {
        let items = vec![
            (1, 100),
            (1, 200),
            (1, 300),
            (2, 400),
            (1, 500),
            (2, 600),
            (1, 700),
        ];

        let lru: Lru<i32, i32> = items.into_iter().collect();

        assert_eq!(lru.len(), 2);
        assert_eq!(lru.capacity(), 2);

        assert_eq!(lru.peek(&1), Some(&700));
        assert_eq!(lru.peek(&2), Some(&600));

        assert_eq!(lru.tail(), Some((&2, &600)));
    }

    #[test]
    fn test_lru_retain_basic() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");
        lru.insert(4, "four");
        lru.insert(5, "five");

        lru.retain(|&k, _| k % 2 == 0);

        assert_eq!(lru.len(), 2);
        assert!(!lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(!lru.contains_key(&3));
        assert!(lru.contains_key(&4));
        assert!(!lru.contains_key(&5));
    }

    #[test]
    fn test_lru_retain_with_recency_considerations() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");
        lru.insert(4, "four");
        lru.insert(5, "five");

        lru.get(&1);
        lru.get(&3);
        lru.get(&2);

        lru.retain(|&k, v| k <= 3 && v.len() > 3);

        assert_eq!(lru.len(), 1);
        assert!(!lru.contains_key(&1));
        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&3));
        assert!(!lru.contains_key(&4));
        assert!(!lru.contains_key(&5));
    }

    #[test]
    fn test_lru_retain_all() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        lru.retain(|_, _| true);

        assert_eq!(lru.len(), 3);
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&3));
    }

    #[test]
    fn test_lru_retain_none() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        lru.retain(|_, _| false);

        assert_eq!(lru.len(), 0);
        assert!(lru.is_empty());
        assert!(!lru.contains_key(&1));
        assert!(!lru.contains_key(&2));
        assert!(!lru.contains_key(&3));
    }

    #[test]
    fn test_lru_retain_empty_cache() {
        let mut lru = Lru::<i32, &str>::new(NonZeroUsize::new(3).unwrap());

        lru.retain(|_, _| true);

        assert_eq!(lru.len(), 0);
        assert!(lru.is_empty());
    }

    #[test]
    fn test_lru_retain_single_item() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");

        lru.retain(|&k, _| k == 1);

        assert_eq!(lru.len(), 1);
        assert!(lru.contains_key(&1));

        lru.retain(|&k, _| k != 1);

        assert_eq!(lru.len(), 0);
        assert!(!lru.contains_key(&1));
    }

    #[test]
    fn test_lru_retain_recency_preservation() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        lru.get(&1);
        lru.get(&2);
        lru.get(&3);

        lru.retain(|&k, _| k == 1 || k == 3);

        assert_eq!(lru.len(), 2);
        assert!(lru.contains_key(&1));
        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&3));

        lru.insert(4, "four");
        lru.insert(5, "five");
        lru.insert(6, "six");
        lru.insert(7, "seven");

        assert!(!lru.contains_key(&1));
        assert!(lru.contains_key(&3));
        assert!(lru.contains_key(&4));
        assert!(lru.contains_key(&5));
        assert!(lru.contains_key(&6));
        assert!(lru.contains_key(&7));
    }

    #[test]
    fn test_lru_retain_with_mutable_values() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());

        lru.insert(1, String::from("short"));
        lru.insert(2, String::from("medium"));
        lru.insert(3, String::from("very_long_string"));
        lru.insert(4, String::from("tiny"));
        lru.insert(5, String::from("another_long_string"));

        lru.retain(|_, v| v.len() > 5);

        assert_eq!(lru.len(), 3);
        assert!(!lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&3));
        assert!(!lru.contains_key(&4));
        assert!(lru.contains_key(&5));
    }

    #[test]
    fn test_lru_retain_stress_test() {
        let mut lru = Lru::new(NonZeroUsize::new(100).unwrap());

        for i in 0..100 {
            lru.insert(i, i * 10);

            if i % 3 == 0 {
                lru.get(&i);
            }
        }

        assert_eq!(lru.len(), 100);

        lru.retain(|&k, _| k % 3 == 0);

        let expected_len = (0..100).filter(|&x| x % 3 == 0).count();
        assert_eq!(lru.len(), expected_len);

        for i in 0..100 {
            if i % 3 == 0 {
                assert!(lru.contains_key(&i));
            } else {
                assert!(!lru.contains_key(&i));
            }
        }
    }

    #[test]
    fn test_lru_retain_after_operations() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");
        lru.insert(4, "four");
        lru.insert(5, "five");

        lru.get(&1);
        lru.get(&3);
        lru.remove(&2);
        lru.get(&4);

        assert_eq!(lru.len(), 4);

        lru.retain(|&k, _| k > 2);

        assert_eq!(lru.len(), 3);
        assert!(!lru.contains_key(&1));
        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&3));
        assert!(lru.contains_key(&4));
        assert!(lru.contains_key(&5));

        lru.insert(6, "six");
        lru.get(&3);
        assert_eq!(lru.len(), 4);
    }

    #[test]
    fn test_lru_retain_linked_list_consistency() {
        let mut lru = Lru::new(NonZeroUsize::new(10).unwrap());

        for i in 1..=10 {
            lru.insert(i, format!("value_{}", i));

            if i % 2 == 0 {
                lru.get(&i);
            }
        }

        lru.retain(|&k, _| k % 2 == 0);

        assert_eq!(lru.len(), 5);
        for i in 1..=10 {
            if i % 2 == 0 {
                assert!(lru.contains_key(&i));
            } else {
                assert!(!lru.contains_key(&i));
            }
        }

        lru.insert(11, "new_item".to_string());
        assert_eq!(lru.len(), 6);
    }

    #[test]
    fn test_lru_retain_with_tail_tracking() {
        let mut lru = Lru::new(NonZeroUsize::new(4).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");
        lru.insert(4, "four");

        lru.get(&2);
        lru.get(&4);
        lru.get(&1);

        assert_eq!(lru.tail(), Some((&3, &"three")));

        lru.retain(|&k, _| k != 3);

        assert_eq!(lru.len(), 3);
        assert!(lru.contains_key(&1));
        assert!(lru.contains_key(&2));
        assert!(!lru.contains_key(&3));
        assert!(lru.contains_key(&4));

        assert_eq!(lru.tail(), Some((&2, &"two")));
    }

    #[test]
    fn test_lru_retain_order_preservation() {
        let mut lru = Lru::new(NonZeroUsize::new(6).unwrap());

        for i in 1..=6 {
            lru.insert(i, i * 10);
        }

        lru.get(&2);
        lru.get(&4);
        lru.get(&6);

        lru.retain(|&k, _| k % 2 == 0);

        assert_eq!(lru.len(), 3);
        assert!(lru.contains_key(&2));
        assert!(lru.contains_key(&4));
        assert!(lru.contains_key(&6));

        assert_eq!(lru.tail(), Some((&2, &20)));

        lru.insert(8, 80);
        lru.insert(10, 100);
        lru.insert(12, 120);
        lru.insert(14, 140);

        assert!(!lru.contains_key(&2));
        assert!(lru.contains_key(&4));
        assert!(lru.contains_key(&6));
        assert!(lru.contains_key(&8));
        assert!(lru.contains_key(&10));
        assert!(lru.contains_key(&12));
        assert!(lru.contains_key(&14));
    }
}
