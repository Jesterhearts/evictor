//! Most Recently Used (MRU) cache implementation.
//!
//! This module provides the MRU eviction policy, which removes the most
//! recently accessed entry when the cache reaches capacity.

use std::hash::Hash;

use crate::{
    EntryValue,
    Metadata,
    Policy,
    RandomState,
    private,
};

/// MRU cache eviction policy implementation.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct MruPolicy;

impl private::Sealed for MruPolicy {}

/// Metadata for MRU policy tracking head, tail, and aging counter.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
#[derive(Default)]
pub struct MruMetadata {
    pub head: usize,
    pub tail: usize,
}

impl private::Sealed for MruMetadata {}

impl Metadata for MruMetadata {
    fn tail_index(&self) -> usize {
        self.tail
    }
}

/// MRU cache entry containing value and linked list pointers.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct MruEntry<V> {
    value: V,
    next: Option<usize>,
    prev: Option<usize>,
}

impl<T> private::Sealed for MruEntry<T> {}

impl<T> EntryValue<T> for MruEntry<T> {
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

impl<T> Policy<T> for MruPolicy {
    type EntryType = MruEntry<T>;
    type MetadataType = MruMetadata;

    fn touch_entry(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut indexmap::IndexMap<impl Hash + Eq, Self::EntryType, RandomState>,
    ) -> usize {
        if index >= queue.len() {
            return index;
        }

        // Move the accessed entry to the tail of the queue
        let old_tail = metadata.tail;
        if old_tail == index {
            return index;
        }

        metadata.tail = index;
        let old_prev = queue[index].prev;
        let old_next = queue[index].next;
        queue[index].prev = None;
        queue[index].next = Some(old_tail);

        if metadata.head == index {
            metadata.head = old_prev.unwrap_or_default();
        }

        queue[old_tail].prev = Some(index);

        if let Some(prev) = old_prev {
            queue[prev].next = old_next;
        }

        if let Some(next) = old_next {
            queue[next].prev = old_prev;
        }

        index
    }

    fn iter_removed_pairs_in_insertion_order<K>(
        pairs: Vec<(K, Self::EntryType)>,
    ) -> impl Iterator<Item = (K, Self::EntryType)> {
        pairs.into_iter().rev()
    }

    fn swap_remove_entry<K: Hash + Eq>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut indexmap::IndexMap<K, Self::EntryType, RandomState>,
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

    use crate::Mru;

    #[test]
    fn test_mru_trivial() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert("a", 1);
        mru.insert("b", 2);
        mru.insert("c", 3);

        assert_eq!(mru.get(&"a"), Some(&1));
        assert_eq!(mru.get(&"b"), Some(&2));
        assert_eq!(mru.get(&"c"), Some(&3));

        mru.get(&"a");
        mru.insert("d", 4);

        assert_eq!(mru.get(&"a"), None);
        assert_eq!(mru.get(&"c"), Some(&3));
    }

    #[test]
    fn test_mru_trivial_get_or_insert() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert("a", 1);
        mru.insert("b", 2);
        mru.insert("c", 3);

        assert_eq!(mru.get_or_insert_with("a", |_| 10), &1);
        assert_eq!(mru.get_or_insert_with("d", |_| 4), &4);
        assert_eq!(mru.get(&"a"), None);
        assert_eq!(mru.get(&"c"), Some(&3));
    }

    #[test]
    fn test_mru_eviction_order() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        mru.insert(4, 40);

        assert_eq!(mru.get(&1), Some(&10));
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), None);
        assert_eq!(mru.get(&4), Some(&40));
    }

    #[test]
    fn test_mru_access_makes_evictable() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        mru.get(&1);

        mru.insert(4, 40);

        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), Some(&30));
        assert_eq!(mru.get(&4), Some(&40));
    }

    #[test]
    fn test_mru_multiple_accesses() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);

        mru.get(&1);
        mru.get(&1);
        mru.get(&1);

        mru.insert(3, 30);

        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), Some(&30));
    }

    #[test]
    fn test_mru_update_existing_key() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);

        mru.insert(1, 100);

        mru.insert(3, 30);

        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), Some(&30));
    }

    #[test]
    fn test_mru_single_capacity() {
        let mut mru = Mru::new(NonZeroUsize::new(1).unwrap());

        mru.insert(1, 10);
        assert_eq!(mru.get(&1), Some(&10));

        mru.insert(2, 20);
        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.get(&2), Some(&20));

        mru.get(&2);
        mru.insert(3, 30);
        assert_eq!(mru.get(&2), None);
        assert_eq!(mru.get(&3), Some(&30));
    }

    #[test]
    fn test_mru_complex_access_pattern() {
        let mut mru = Mru::new(NonZeroUsize::new(4).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);
        mru.insert(4, 40);

        mru.get(&2);
        mru.get(&1);
        mru.get(&3);

        mru.insert(5, 50);

        assert_eq!(mru.get(&1), Some(&10));
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), None);
        assert_eq!(mru.get(&4), Some(&40));
        assert_eq!(mru.get(&5), Some(&50));
    }

    #[test]
    fn test_mru_interleaved_operations() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.get(&1);
        mru.insert(3, 30);
        mru.get(&2);
        mru.insert(4, 40);

        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.get(&2), None);
        assert_eq!(mru.get(&3), Some(&30));
        assert_eq!(mru.get(&4), Some(&40));
    }

    #[test]
    fn test_mru_get_or_insert_with_eviction() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);

        mru.get(&1);

        let value = mru.get_or_insert_with(3, |_| 30);
        assert_eq!(value, &30);

        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), Some(&30));
    }

    #[test]
    fn test_mru_get_or_insert_existing_key() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        let value = mru.get_or_insert_with(1, |_| 999);
        assert_eq!(value, &10);

        mru.insert(4, 40);

        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), Some(&30));
        assert_eq!(mru.get(&4), Some(&40));
    }

    #[test]
    fn test_mru_sequential_pattern() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        for i in 1..=10 {
            mru.insert(i, i * 10);
        }

        assert_eq!(mru.get(&10), Some(&100));

        let mut count = 0;
        for i in 1..=10 {
            if mru.get(&i).is_some() {
                count += 1;
            }
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_mru_no_access_pattern() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        mru.insert(4, 40);

        assert_eq!(mru.get(&1), Some(&10));
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), None);
        assert_eq!(mru.get(&4), Some(&40));
    }

    #[test]
    fn test_mru_alternating_access() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        mru.get(&1);
        mru.get(&2);
        mru.get(&1);
        mru.get(&2);

        mru.insert(4, 40);

        assert_eq!(mru.get(&1), Some(&10));
        assert_eq!(mru.get(&2), None);
        assert_eq!(mru.get(&3), Some(&30));
        assert_eq!(mru.get(&4), Some(&40));
    }

    #[test]
    fn test_mru_repeated_access_single_key() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);

        for _ in 0..100 {
            mru.get(&1);
        }

        mru.insert(3, 30);

        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), Some(&30));
    }

    #[test]
    fn test_mru_empty_cache() {
        let mut mru = Mru::<i32, i32>::new(NonZeroUsize::new(3).unwrap());

        assert!(mru.is_empty());
        assert_eq!(mru.len(), 0);
        assert_eq!(mru.capacity(), 3);
        assert_eq!(mru.get(&1), None);
        assert_eq!(mru.peek(&1), None);
        assert_eq!(mru.remove(&1), None);
        assert_eq!(mru.pop(), None);
        assert!(mru.tail().is_none());
        assert!(!mru.contains_key(&1));
    }

    #[test]
    fn test_mru_capacity_constraints() {
        let mru = Mru::<i32, i32>::new(NonZeroUsize::new(5).unwrap());
        assert_eq!(mru.capacity(), 5);

        let mru = Mru::<i32, i32>::new(NonZeroUsize::new(1).unwrap());
        assert_eq!(mru.capacity(), 1);

        let mru = Mru::<i32, i32>::new(NonZeroUsize::new(100).unwrap());
        assert_eq!(mru.capacity(), 100);
    }

    #[test]
    fn test_mru_peek_no_side_effects() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        assert_eq!(mru.peek(&3), Some(&30));

        mru.insert(4, 40);

        assert_eq!(mru.get(&1), Some(&10));
        assert_eq!(mru.get(&2), Some(&20));
        assert_eq!(mru.get(&3), None);
        assert_eq!(mru.get(&4), Some(&40));
    }

    #[test]
    fn test_mru_contains_key() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        assert!(!mru.contains_key(&1));

        mru.insert(1, 10);
        assert!(mru.contains_key(&1));
        assert!(!mru.contains_key(&2));

        mru.insert(2, 20);
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));

        mru.insert(3, 30);
        assert!(mru.contains_key(&1));
        assert!(!mru.contains_key(&2));
        assert!(mru.contains_key(&3));
    }

    #[test]
    fn test_mru_remove() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        assert_eq!(mru.remove(&2), Some(20));
        assert_eq!(mru.len(), 2);
        assert!(!mru.contains_key(&2));
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&3));

        assert_eq!(mru.remove(&2), None);
        assert_eq!(mru.len(), 2);

        mru.insert(4, 40);
        assert_eq!(mru.len(), 3);
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&3));
        assert!(mru.contains_key(&4));
    }

    #[test]
    fn test_mru_pop() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        let popped = mru.pop();
        assert!(popped.is_some());
        assert_eq!(mru.len(), 2);

        let popped = mru.pop();
        assert!(popped.is_some());
        assert_eq!(mru.len(), 1);

        let popped = mru.pop();
        assert!(popped.is_some());
        assert_eq!(mru.len(), 0);
        assert!(mru.is_empty());

        let popped = mru.pop();
        assert_eq!(popped, None);
    }

    #[test]
    fn test_mru_tail() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        assert!(mru.tail().is_none());

        mru.insert(1, 10);
        assert_eq!(mru.tail(), Some((&1, &10)));

        mru.insert(2, 20);
        assert_eq!(mru.tail(), Some((&2, &20)));

        mru.insert(3, 30);
        assert_eq!(mru.tail(), Some((&3, &30)));

        mru.get(&2);

        assert_eq!(mru.tail(), Some((&2, &20)));
    }

    #[test]
    fn test_mru_clear() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        assert_eq!(mru.len(), 3);
        assert!(!mru.is_empty());

        mru.clear();

        assert_eq!(mru.len(), 0);
        assert!(mru.is_empty());
        assert_eq!(mru.capacity(), 3);
        assert!(!mru.contains_key(&1));
        assert!(!mru.contains_key(&2));
        assert!(!mru.contains_key(&3));
        assert!(mru.tail().is_none());
    }

    #[test]
    fn test_mru_mutable_access() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert(1, String::from("hello"));
        mru.insert(2, String::from("world"));
        mru.insert(3, String::from("test"));

        if let Some(val) = mru.get_mut(&1) {
            val.push_str(" modified");
        }

        assert_eq!(mru.get(&1), Some(&String::from("hello modified")));

        mru.insert(4, String::from("new"));

        assert!(!mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&3));
        assert!(mru.contains_key(&4));
    }

    #[test]
    fn test_mru_insert_mut() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        let val = mru.insert_mut(1, String::from("test"));
        val.push_str(" modified");

        assert_eq!(mru.get(&1), Some(&String::from("test modified")));

        let val = mru.insert_mut(1, String::from("replaced"));
        val.push_str(" again");

        assert_eq!(mru.get(&1), Some(&String::from("replaced again")));
    }

    #[test]
    fn test_mru_get_or_insert_with_mut() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());
        mru.insert(1, String::from("existing"));

        let val = mru.get_or_insert_with_mut(1, |_| String::from("new"));
        val.push_str(" modified");

        assert_eq!(mru.get(&1), Some(&String::from("existing modified")));

        let val = mru.get_or_insert_with_mut(2, |_| String::from("created"));
        val.push_str(" also modified");

        assert_eq!(mru.get(&2), Some(&String::from("created also modified")));

        mru.insert(3, String::from("third"));
        assert!(mru.contains_key(&1));
        assert!(!mru.contains_key(&2));
        assert!(mru.contains_key(&3));
    }

    #[test]
    fn test_mru_extend() {
        let mut mru = Mru::new(NonZeroUsize::new(5).unwrap());
        mru.insert(1, 10);

        let items = vec![(2, 20), (3, 30), (4, 40)];
        mru.extend(items);

        assert_eq!(mru.len(), 4);
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&3));
        assert!(mru.contains_key(&4));
    }

    #[test]
    fn test_mru_from_iterator() {
        let items = vec![(1, 10), (2, 20), (3, 30)];
        let mru: Mru<i32, i32> = items.into_iter().collect();

        assert_eq!(mru.len(), 3);
        assert_eq!(mru.capacity(), 3);
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&3));
    }

    #[test]
    fn test_mru_from_iterator_with_eviction() {
        let items = vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)];
        let mru: Mru<i32, i32> = items.into_iter().collect();

        assert_eq!(mru.len(), 5);
        assert_eq!(mru.capacity(), 5);

        for i in 1..=5 {
            assert!(mru.contains_key(&i));
        }
    }

    #[test]
    fn test_mru_shrink_to_fit() {
        let mut mru = Mru::new(NonZeroUsize::new(10).unwrap());
        mru.insert(1, 10);
        mru.insert(2, 20);

        mru.shrink_to_fit();

        assert_eq!(mru.len(), 2);
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
    }

    #[test]
    fn test_mru_eviction_policy_consistency() {
        let mut mru = Mru::new(NonZeroUsize::new(4).unwrap());

        mru.insert(1, "first");
        mru.insert(2, "second");
        mru.insert(3, "third");
        mru.insert(4, "fourth");

        mru.get(&2);
        mru.get(&1);
        mru.get(&3);

        mru.insert(5, "fifth");

        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(!mru.contains_key(&3));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&5));

        mru.get(&1);

        mru.insert(6, "sixth");

        assert!(!mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&5));
        assert!(mru.contains_key(&6));
    }

    #[test]
    fn test_mru_stress_operations() {
        let mut mru = Mru::new(NonZeroUsize::new(10).unwrap());

        for i in 0..20 {
            mru.insert(i, i * 10);
        }

        assert_eq!(mru.len(), 10);

        for i in 10..20 {
            mru.get(&i);
            if i % 3 == 0 {
                mru.remove(&(i - 5));
            }
        }

        assert!(mru.len() <= 10);
    }

    #[test]
    fn test_mru_boundary_conditions() {
        let mut mru = Mru::new(NonZeroUsize::new(1000).unwrap());
        for i in 0..500 {
            mru.insert(i, i);
        }
        assert_eq!(mru.len(), 500);

        for i in 500..1000 {
            mru.insert(i, i);
        }
        assert_eq!(mru.len(), 1000);

        mru.insert(1000, 1000);
        assert_eq!(mru.len(), 1000);
        assert!(mru.contains_key(&1000));
    }

    #[test]
    fn test_mru_consistent_state_after_operations() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 100);
        mru.insert(2, 200);
        assert_eq!(mru.len(), 2);

        mru.get(&1);
        mru.insert(3, 300);
        assert_eq!(mru.len(), 3);

        mru.remove(&2);
        assert_eq!(mru.len(), 2);
        assert!(!mru.contains_key(&2));

        mru.insert(4, 400);
        mru.insert(5, 500);
        assert_eq!(mru.len(), 3);

        assert!(mru.tail().is_some());
        let mut count = 0;
        for i in 1..=5 {
            if mru.contains_key(&i) {
                count += 1;
            }
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_mru_key_value_types() {
        let mut mru_str = Mru::new(NonZeroUsize::new(2).unwrap());
        mru_str.insert("key1", 1);
        mru_str.insert("key2", 2);
        assert!(mru_str.contains_key(&"key1"));

        let mut mru_vec = Mru::new(NonZeroUsize::new(2).unwrap());
        mru_vec.insert(1, vec![1, 2, 3]);
        mru_vec.insert(2, vec![4, 5, 6]);
        assert_eq!(mru_vec.get(&1), Some(&vec![1, 2, 3]));
    }

    #[test]
    fn test_mru_insertion_order_vs_access_order() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        mru.insert(4, 40);

        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(!mru.contains_key(&3));
        assert!(mru.contains_key(&4));

        mru.get(&1);
        mru.insert(5, 50);

        assert!(!mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&5));
    }

    #[test]
    fn test_mru_recency_tracking() {
        let mut mru = Mru::new(NonZeroUsize::new(4).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);
        mru.insert(4, 40);

        mru.get(&2);
        mru.get(&4);
        mru.get(&1);

        mru.insert(5, 50);

        assert!(!mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&3));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&5));

        mru.get(&3);
        mru.insert(6, 60);

        assert!(mru.contains_key(&2));
        assert!(!mru.contains_key(&3));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&5));
        assert!(mru.contains_key(&6));
    }

    #[test]
    fn test_mru_from_iterator_overlapping_keys() {
        let items = vec![
            (1, "first"),
            (2, "second"),
            (1, "updated_first"),
            (3, "third"),
            (2, "updated_second"),
            (1, "final_first"),
        ];

        let mru: Mru<i32, &str> = items.into_iter().collect();

        assert_eq!(mru.len(), 3);
        assert_eq!(mru.capacity(), 3);

        assert_eq!(mru.peek(&1), Some(&"final_first"));
        assert_eq!(mru.peek(&2), Some(&"updated_second"));
        assert_eq!(mru.peek(&3), Some(&"third"));

        assert_eq!(mru.tail(), Some((&1, &"final_first")));
    }

    #[test]
    fn test_mru_from_iterator_overlapping_with_eviction() {
        let items = vec![
            (1, "one"),
            (2, "two"),
            (3, "three"),
            (4, "four"),
            (5, "five"),
            (1, "updated_one"),
            (6, "six"),
        ];

        let mru: Mru<i32, &str> = items.into_iter().collect();

        assert_eq!(mru.len(), 6);
        assert_eq!(mru.capacity(), 6);

        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&3));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&5));
        assert!(mru.contains_key(&6));

        assert_eq!(mru.peek(&1), Some(&"updated_one"));
    }

    #[test]
    fn test_mru_from_iterator_many_duplicates() {
        let items = vec![
            (1, 100),
            (1, 200),
            (1, 300),
            (2, 400),
            (1, 500),
            (2, 600),
            (1, 700),
        ];

        let mru: Mru<i32, i32> = items.into_iter().collect();

        assert_eq!(mru.len(), 2);
        assert_eq!(mru.capacity(), 2);

        assert_eq!(mru.peek(&1), Some(&700));
        assert_eq!(mru.peek(&2), Some(&600));

        assert_eq!(mru.tail(), Some((&1, &700)));
    }

    #[test]
    fn test_mru_from_iterator_access_order_behavior() {
        let items = vec![
            (1, "one"),
            (2, "two"),
            (3, "three"),
            (2, "updated_two"),
            (4, "four"),
            (1, "updated_one"),
        ];

        let mut mru: Mru<i32, &str> = items.into_iter().collect();

        assert_eq!(mru.len(), 4);

        assert_eq!(mru.tail(), Some((&1, &"updated_one")));

        mru.insert(5, "five");

        assert!(!mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&3));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&5));
    }

    #[test]
    fn test_mru_retain_basic() {
        let mut mru = Mru::new(NonZeroUsize::new(5).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");
        mru.insert(4, "four");
        mru.insert(5, "five");

        mru.retain(|&k, _| k % 2 == 0);

        assert_eq!(mru.len(), 2);
        assert!(!mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(!mru.contains_key(&3));
        assert!(mru.contains_key(&4));
        assert!(!mru.contains_key(&5));
    }

    #[test]
    fn test_mru_retain_with_recency_considerations() {
        let mut mru = Mru::new(NonZeroUsize::new(5).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");
        mru.insert(4, "four");
        mru.insert(5, "five");

        mru.get(&1);
        mru.get(&3);
        mru.get(&2);

        mru.retain(|&k, v| k <= 3 && v.len() > 3);

        assert_eq!(mru.len(), 1);
        assert!(!mru.contains_key(&1));
        assert!(!mru.contains_key(&2));
        assert!(mru.contains_key(&3));
        assert!(!mru.contains_key(&4));
        assert!(!mru.contains_key(&5));
    }

    #[test]
    fn test_mru_retain_all() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        mru.retain(|_, _| true);

        assert_eq!(mru.len(), 3);
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&3));
    }

    #[test]
    fn test_mru_retain_none() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        mru.retain(|_, _| false);

        assert_eq!(mru.len(), 0);
        assert!(mru.is_empty());
        assert!(!mru.contains_key(&1));
        assert!(!mru.contains_key(&2));
        assert!(!mru.contains_key(&3));
    }

    #[test]
    fn test_mru_retain_empty_cache() {
        let mut mru = Mru::<i32, &str>::new(NonZeroUsize::new(3).unwrap());

        mru.retain(|_, _| true);

        assert_eq!(mru.len(), 0);
        assert!(mru.is_empty());
    }

    #[test]
    fn test_mru_retain_single_item() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");

        mru.retain(|&k, _| k == 1);

        assert_eq!(mru.len(), 1);
        assert!(mru.contains_key(&1));

        mru.retain(|&k, _| k != 1);

        assert_eq!(mru.len(), 0);
        assert!(!mru.contains_key(&1));
    }

    #[test]
    fn test_mru_retain_recency_preservation() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        mru.retain(|&k, _| k == 1 || k == 3);

        assert_eq!(mru.len(), 2);
        assert!(mru.contains_key(&1));
        assert!(!mru.contains_key(&2));
        assert!(mru.contains_key(&3));

        assert_eq!(mru.tail(), Some((&3, &"three")));
        mru.insert(4, "four");

        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&3));
        assert!(mru.contains_key(&4));
        assert_eq!(mru.len(), 3);
    }

    #[test]
    fn test_mru_retain_stress_test() {
        let mut mru = Mru::new(NonZeroUsize::new(100).unwrap());

        for i in 0..100 {
            mru.insert(i, i * 10);

            if i % 3 == 0 {
                mru.get(&i);
            }
        }

        assert_eq!(mru.len(), 100);

        mru.retain(|&k, _| k % 3 == 0);

        let expected_len = (0..100).filter(|&x| x % 3 == 0).count();
        assert_eq!(mru.len(), expected_len);

        for i in 0..100 {
            if i % 3 == 0 {
                assert!(mru.contains_key(&i));
            } else {
                assert!(!mru.contains_key(&i));
            }
        }
    }

    #[test]
    fn test_mru_retain_after_operations() {
        let mut mru = Mru::new(NonZeroUsize::new(5).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");
        mru.insert(4, "four");
        mru.insert(5, "five");

        mru.get(&1);
        mru.get(&3);
        mru.remove(&2);
        mru.get(&4);

        assert_eq!(mru.len(), 4);

        mru.retain(|&k, _| k > 2);

        assert_eq!(mru.len(), 3);
        assert!(!mru.contains_key(&1));
        assert!(!mru.contains_key(&2));
        assert!(mru.contains_key(&3));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&5));

        mru.insert(6, "six");
        mru.get(&3);
        assert_eq!(mru.len(), 4);
    }

    #[test]
    fn test_mru_retain_linked_list_consistency() {
        let mut mru = Mru::new(NonZeroUsize::new(10).unwrap());

        for i in 1..=10 {
            mru.insert(i, format!("value_{}", i));

            if i % 2 == 0 {
                mru.get(&i);
            }
        }

        mru.retain(|&k, _| k % 2 == 0);

        assert_eq!(mru.len(), 5);
        for i in 1..=10 {
            if i % 2 == 0 {
                assert!(mru.contains_key(&i));
            } else {
                assert!(!mru.contains_key(&i));
            }
        }

        mru.insert(12, "new_item".to_string());
        assert_eq!(mru.len(), 6);
    }

    #[test]
    fn test_mru_retain_with_tail_tracking() {
        let mut mru = Mru::new(NonZeroUsize::new(4).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");
        mru.insert(4, "four");

        mru.get(&2);
        mru.get(&4);
        mru.get(&1);

        assert_eq!(mru.tail(), Some((&1, &"one")));

        mru.retain(|&k, _| k != 3);

        assert_eq!(mru.len(), 3);
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(!mru.contains_key(&3));
        assert!(mru.contains_key(&4));

        assert_eq!(mru.tail(), Some((&1, &"one")));
    }

    #[test]
    fn test_mru_retain_eviction_behavior() {
        let mut mru = Mru::new(NonZeroUsize::new(4).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");
        mru.insert(4, "four");

        mru.get(&1);
        mru.get(&3);

        mru.retain(|&k, _| k != 4);

        assert_eq!(mru.len(), 3);
        assert!(mru.contains_key(&1));
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&3));
        assert!(!mru.contains_key(&4));

        assert_eq!(mru.tail(), Some((&3, &"three")));

        mru.insert(5, "five");
        mru.insert(6, "six");

        assert_eq!(mru.len(), 4);

        mru.insert(7, "seven");
        assert_eq!(mru.len(), 4);
    }

    #[test]
    fn test_mru_retain_order_preservation() {
        let mut mru = Mru::new(NonZeroUsize::new(6).unwrap());

        for i in 1..=6 {
            mru.insert(i, i * 10);
        }

        mru.get(&2);
        mru.get(&4);
        mru.get(&6);

        mru.retain(|&k, _| k % 2 == 0);

        assert_eq!(mru.len(), 3);
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&6));

        assert_eq!(mru.tail(), Some((&6, &60)));

        mru.insert(8, 80);
        mru.insert(10, 100);
        mru.insert(12, 120);

        assert_eq!(mru.len(), 6);
        assert!(mru.contains_key(&2));
        assert!(mru.contains_key(&4));
        assert!(mru.contains_key(&6));
        assert!(mru.contains_key(&8));
        assert!(mru.contains_key(&10));
        assert!(mru.contains_key(&12));
    }
}
