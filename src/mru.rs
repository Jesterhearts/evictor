use std::hash::Hash;

use crate::{
    EntryValue,
    Metadata,
    Policy,
    RandomState,
    private,
    utils::{
        impl_ll_iters,
        swap_remove_ll_entry,
    },
};

#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct MruPolicy;

impl private::Sealed for MruPolicy {}

#[derive(Debug, Clone, Copy, Default)]
#[doc(hidden)]
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
    type IntoIter<K> = IntoIter<K, T>;
    type MetadataType = MruMetadata;

    fn touch_entry<K>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut indexmap::IndexMap<K, Self::EntryType, RandomState>,
    ) -> usize {
        if index >= queue.len() {
            return index;
        }

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

    fn swap_remove_entry<K: Hash + Eq>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut indexmap::IndexMap<K, Self::EntryType, RandomState>,
    ) -> Option<(K, Self::EntryType)> {
        swap_remove_ll_entry!(index, metadata, queue)
    }

    fn iter<'q, K>(
        metadata: &'q Self::MetadataType,
        queue: &'q indexmap::IndexMap<K, Self::EntryType, RandomState>,
    ) -> impl Iterator<Item = (&'q K, &'q T)>
    where
        T: 'q,
    {
        Iter {
            queue,
            index: Some(metadata.tail),
        }
    }

    fn into_iter<K>(
        metadata: Self::MetadataType,
        queue: indexmap::IndexMap<K, Self::EntryType, RandomState>,
    ) -> IntoIter<K, T> {
        IntoIter {
            queue: queue.into_iter().map(Some).collect(),
            index: Some(metadata.tail),
        }
    }

    fn into_entries<K>(
        metadata: Self::MetadataType,
        queue: indexmap::IndexMap<K, Self::EntryType, RandomState>,
    ) -> impl Iterator<Item = (K, Self::EntryType)> {
        IntoEntriesIter {
            queue: queue.into_iter().map(Some).collect(),
            index: Some(metadata.tail),
        }
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (2, 20), (1, 10)]
        );
    }

    #[test]
    fn test_mru_access_makes_evictable() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        mru.get(&1);

        mru.insert(4, 40);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (3, 30), (2, 20)]
        );
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

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(3, 30), (2, 20)]);
    }

    #[test]
    fn test_mru_update_existing_key() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);

        mru.insert(1, 100);

        mru.insert(3, 30);

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(3, 30), (2, 20)]);
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

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(3, 30)]);
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(5, 50), (1, 10), (2, 20), (4, 40)]
        );
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

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(4, 40), (3, 30)]);
    }

    #[test]
    fn test_mru_get_or_insert_with_eviction() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);

        mru.get(&1);

        let value = mru.get_or_insert_with(3, |_| 30);
        assert_eq!(value, &30);

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(3, 30), (2, 20)]);
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (3, 30), (2, 20)]
        );
    }

    #[test]
    fn test_mru_sequential_pattern() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        for i in 1..=10 {
            mru.insert(i, i * 10);
        }

        // MRU cache should contain the 3 most recently inserted items
        // but since we accessed item 10, it's not in the expected order
        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(10, 100), (2, 20), (1, 10)]
        );
    }

    #[test]
    fn test_mru_no_access_pattern() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, 10);
        mru.insert(2, 20);
        mru.insert(3, 30);

        mru.insert(4, 40);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (2, 20), (1, 10)]
        );
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (1, 10), (3, 30)]
        );
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

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(3, 30), (2, 20)]);
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (2, 20), (1, 10)]
        );
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

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(3, 30), (1, 10)]);
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (3, 30), (1, 10)]
        );
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
        assert!(mru.tail().is_none());

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    fn test_mru_mutable_access() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert(1, "hello".to_string());
        mru.insert(2, "world".to_string());
        mru.insert(3, "test".to_string());

        if let Some(val) = mru.get_mut(&1) {
            val.push_str(" modified");
        }

        assert_eq!(mru.get(&1), Some(&"hello modified".to_string()));

        mru.insert(4, "new".to_string());

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [
                (4, "new".to_string()),
                (3, "test".to_string()),
                (2, "world".to_string())
            ]
        );
    }

    #[test]
    fn test_mru_insert_mut() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        let val = mru.insert_mut(1, "test".to_string());
        val.push_str(" modified");

        assert_eq!(mru.get(&1), Some(&"test modified".to_string()));

        let val = mru.insert_mut(1, "replaced".to_string());
        val.push_str(" again");

        assert_eq!(mru.get(&1), Some(&"replaced again".to_string()));
    }

    #[test]
    fn test_mru_get_or_insert_with_mut() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());
        mru.insert(1, "existing".to_string());

        let val = mru.get_or_insert_with_mut(1, |_| "new".to_string());
        val.push_str(" modified");

        assert_eq!(mru.get(&1), Some(&"existing modified".to_string()));

        let val = mru.get_or_insert_with_mut(2, |_| "created".to_string());
        val.push_str(" also modified");

        assert_eq!(mru.get(&2), Some(&"created also modified".to_string()));

        mru.insert(3, "third".to_string());

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [
                (3, "third".to_string()),
                (1, "existing modified".to_string())
            ]
        );
    }

    #[test]
    fn test_mru_extend() {
        let mut mru = Mru::new(NonZeroUsize::new(5).unwrap());
        mru.insert(1, 10);

        let items = vec![(2, 20), (3, 30), (4, 40)];
        mru.extend(items);

        assert_eq!(mru.len(), 4);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (3, 30), (2, 20), (1, 10)]
        );
    }

    #[test]
    fn test_mru_from_iterator() {
        let items = vec![(1, 10), (2, 20), (3, 30)];
        let mru: Mru<i32, i32> = items.into_iter().collect();

        assert_eq!(mru.len(), 3);
        assert_eq!(mru.capacity(), 3);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(3, 30), (2, 20), (1, 10)]
        );
    }

    #[test]
    fn test_mru_from_iterator_with_eviction() {
        let items = vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)];
        let mru: Mru<i32, i32> = items.into_iter().collect();

        assert_eq!(mru.len(), 5);
        assert_eq!(mru.capacity(), 5);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(5, 50), (4, 40), (3, 30), (2, 20), (1, 10)]
        );
    }

    #[test]
    fn test_mru_shrink_to_fit() {
        let mut mru = Mru::new(NonZeroUsize::new(10).unwrap());
        mru.insert(1, 10);
        mru.insert(2, 20);

        mru.shrink_to_fit();

        assert_eq!(mru.len(), 2);

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(2, 20), (1, 10)]);
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
        mru.get(&1);
        mru.insert(6, "sixth");

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(6, "sixth"), (5, "fifth"), (2, "second"), (4, "fourth")]
        );
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
        mru.get(&1);
        mru.insert(5, 50);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(5, 50), (4, 40), (2, 20)]
        );
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
        mru.get(&3);
        mru.insert(6, 60);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(6, 60), (5, 50), (4, 40), (2, 20)]
        );
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [
                (6, "six"),
                (1, "updated_one"),
                (5, "five"),
                (4, "four"),
                (3, "three"),
                (2, "two")
            ]
        );
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(5, "five"), (4, "four"), (2, "updated_two"), (3, "three")]
        );
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, "four"), (2, "two")]
        );
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

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), [(3, "three")]);
    }

    #[test]
    fn test_mru_retain_all() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        mru.retain(|_, _| true);

        assert_eq!(mru.len(), 3);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(3, "three"), (2, "two"), (1, "one")]
        );
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

        assert_eq!(mru.into_iter().collect::<Vec<_>>(), []);
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
        assert_eq!(mru.into_iter().collect::<Vec<_>>(), []);
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

        assert_eq!(mru.len(), 3);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(4, "four"), (3, "three"), (1, "one")]
        );
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

        let expected_keys: Vec<_> = (0..100)
            .filter(|&x| x % 3 == 0)
            .map(|x| (x, x * 10))
            .rev()
            .collect();
        assert_eq!(mru.into_iter().collect::<Vec<_>>(), expected_keys);
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

        mru.insert(6, "six");
        mru.get(&3);
        assert_eq!(mru.len(), 4);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(3, "three"), (6, "six"), (4, "four"), (5, "five")]
        );
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

        mru.insert(12, "new_item".to_string());
        assert_eq!(mru.len(), 6);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [
                (12, "new_item".to_string()),
                (10, "value_10".to_string()),
                (8, "value_8".to_string()),
                (6, "value_6".to_string()),
                (4, "value_4".to_string()),
                (2, "value_2".to_string())
            ]
        );
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
        assert_eq!(mru.tail(), Some((&1, &"one")));

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(1, "one"), (4, "four"), (2, "two")]
        );
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

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(12, 120), (10, 100), (8, 80), (6, 60), (4, 40), (2, 20)]
        );
    }

    #[test]
    fn test_mru_iter_empty_cache() {
        let mru = Mru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
        let items: Vec<_> = mru.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_mru_iter_single_item() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());
        mru.insert(1, "one");

        let items: Vec<_> = mru.iter().collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], (&1, &"one"));
    }

    #[test]
    fn test_mru_iter_eviction_order() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(3, "three"), (2, "two"), (1, "one")]
        );
    }

    #[test]
    fn test_mru_iter_after_access() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        mru.get(&1);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(1, "one"), (3, "three"), (2, "two")]
        );
    }

    #[test]
    fn test_mru_iter_complex_access_pattern() {
        let mut mru = Mru::new(NonZeroUsize::new(4).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");
        mru.insert(4, "four");

        mru.get(&2);
        mru.get(&4);
        mru.get(&1);

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(1, "one"), (4, "four"), (2, "two"), (3, "three")]
        );
    }

    #[test]
    fn test_mru_iter_after_update() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        *mru.get_mut(&1).unwrap() = "updated_one";

        let items: Vec<_> = mru.iter().collect();
        assert_eq!(items, [(&1, &"updated_one"), (&3, &"three"), (&2, &"two")]);
    }

    #[test]
    fn test_mru_iter_with_eviction() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        assert_eq!(
            mru.into_iter().collect::<Vec<_>>(),
            [(3, "three"), (1, "one")]
        );
    }

    #[test]
    fn test_mru_iter_eviction_after_access() {
        let mut mru = Mru::new(NonZeroUsize::new(2).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");

        mru.get(&1);

        mru.insert(3, "three");

        let items: Vec<_> = mru.iter().collect();
        assert_eq!(items, [(&3, &"three"), (&2, &"two")]);
    }

    #[test]
    fn test_mru_iter_consistency_with_tail() {
        let mut mru = Mru::new(NonZeroUsize::new(4).unwrap());

        mru.insert(10, "ten");
        mru.insert(20, "twenty");
        mru.insert(30, "thirty");
        mru.insert(40, "forty");

        mru.get(&20);
        mru.get(&30);

        let tail = mru.tail();
        let mut iter_items = mru.iter();
        let first_iter_item = iter_items.next();

        assert_eq!(tail, first_iter_item);
    }

    #[test]
    fn test_mru_iter_multiple_iterations() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        let items1: Vec<_> = mru.iter().collect();
        let items2: Vec<_> = mru.iter().collect();

        assert_eq!(items1, items2);
        assert_eq!(items1.len(), 3);
    }

    #[test]
    fn test_mru_iter_after_remove() {
        let mut mru = Mru::new(NonZeroUsize::new(4).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");
        mru.insert(4, "four");

        mru.get(&2);
        mru.remove(&3);

        let items: Vec<_> = mru.iter().collect();
        assert_eq!(items, [(&2, &"two"), (&4, &"four"), (&1, &"one")]);
    }

    #[test]
    fn test_mru_iter_repeated_access_pattern() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        mru.get(&1);
        mru.get(&1);
        mru.get(&1);

        let items: Vec<_> = mru.iter().collect();
        assert_eq!(items, [(&1, &"one"), (&3, &"three"), (&2, &"two")]);
    }

    #[test]
    fn test_into_iter_matches_iter() {
        let mut mru = Mru::new(NonZeroUsize::new(3).unwrap());

        mru.insert(1, "one");
        mru.insert(2, "two");
        mru.insert(3, "three");

        let items: Vec<_> = mru.iter().map(|(k, v)| (*k, *v)).collect();
        let into_items: Vec<_> = mru.into_iter().collect();

        assert_eq!(items, into_items);
    }

    #[test]
    fn test_mru_peek_mut_no_modification() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
        cache.insert("A", vec![1, 2, 3]);
        cache.insert("B", vec![4, 5, 6]);
        cache.insert("C", vec![7, 8, 9]);

        let original_order: Vec<_> = cache.iter().map(|(k, _)| *k).collect();

        if let Some(entry) = cache.peek_mut(&"A") {
            let _len = entry.len();
        }

        let new_order: Vec<_> = cache.iter().map(|(k, _)| *k).collect();
        assert_eq!(original_order, new_order);
    }

    #[test]
    fn test_mru_peek_mut_with_modification() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
        cache.insert("A", vec![1, 2, 3]);
        cache.insert("B", vec![4, 5, 6]);
        cache.insert("C", vec![7, 8, 9]);

        assert_eq!(cache.tail().unwrap().0, &"C");

        if let Some(mut entry) = cache.peek_mut(&"A") {
            entry.push(4);
        }

        assert_eq!(cache.peek(&"A"), Some(&vec![1, 2, 3, 4]));
        assert_eq!(cache.tail().unwrap().0, &"A");
    }

    #[test]
    fn test_mru_peek_mut_nonexistent_key() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one");
        cache.insert(2, "two");

        assert!(cache.peek_mut(&3).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_mru_peek_mut_empty_cache() {
        let mut cache = Mru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
        assert!(cache.peek_mut(&1).is_none());
    }

    #[test]
    fn test_mru_peek_mut_single_item() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());

        if let Some(mut entry) = cache.peek_mut(&1) {
            entry.push_str("_modified");
        }

        assert_eq!(cache.peek(&1), Some(&"one_modified".to_string()));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.tail().unwrap().0, &1);
    }

    #[test]
    fn test_mru_peek_mut_with_eviction() {
        let mut cache = Mru::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, 10);
        cache.insert(2, 20);

        assert_eq!(cache.tail().unwrap().0, &2);

        if let Some(mut entry) = cache.peek_mut(&1) {
            *entry += 5;
        }

        assert_eq!(cache.tail().unwrap().0, &1);

        cache.insert(3, 30);

        assert!(cache.peek(&1).is_none());
        assert_eq!(cache.peek(&2), Some(&20));
        assert_eq!(cache.peek(&3), Some(&30));
    }

    #[test]
    fn test_mru_peek_mut_preserve_order_on_no_modification() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);

        cache.get(&1);

        let original_tail = cache.tail().map(|(k, _)| *k);

        if let Some(entry) = cache.peek_mut(&2) {
            let _value = *entry;
        }

        let new_tail = cache.tail().map(|(k, _)| *k);
        assert_eq!(original_tail, new_tail);
    }

    #[test]
    fn test_mru_peek_mut_multiple_modifications() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "hello".to_string());
        cache.insert(2, "world".to_string());

        if let Some(mut entry) = cache.peek_mut(&1) {
            entry.push_str(" there");
        }

        if let Some(mut entry) = cache.peek_mut(&1) {
            entry.push_str(" friend");
        }

        assert_eq!(cache.peek(&1), Some(&"hello there friend".to_string()));
        assert_eq!(cache.tail().unwrap().0, &1);
    }

    #[test]
    fn test_mru_peek_mut_ordering_consistency() {
        let mut cache = Mru::new(NonZeroUsize::new(4).unwrap());
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);
        cache.insert(4, 400);

        cache.get(&2);
        cache.get(&1);

        let before_modification: Vec<_> = cache.iter().map(|(k, v)| (*k, *v)).collect();

        if let Some(mut entry) = cache.peek_mut(&3) {
            *entry += 1000;
        }

        let after_modification: Vec<_> = cache.iter().map(|(k, v)| (*k, *v)).collect();

        assert_ne!(before_modification[0].0, after_modification[0].0);
        assert_eq!(after_modification[0].0, 3);
        assert_eq!(cache.peek(&3), Some(&1300));
        assert_eq!(cache.tail().unwrap().0, &3);
    }

    #[test]
    fn test_mru_peek_mut_complex_scenario() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
        cache.insert("first", 1);
        cache.insert("second", 2);
        cache.insert("third", 3);

        cache.get(&"first");

        assert_eq!(cache.tail().unwrap().0, &"first");

        if let Some(mut entry) = cache.peek_mut(&"second") {
            *entry += 100;
        }

        assert_eq!(cache.peek(&"second"), Some(&102));
        assert_eq!(cache.tail().unwrap().0, &"second");

        cache.insert("fourth", 4);

        assert!(!cache.contains_key(&"second"));
        assert!(cache.contains_key(&"first"));
        assert!(cache.contains_key(&"third"));
        assert!(cache.contains_key(&"fourth"));
    }

    #[test]
    fn test_mru_peek_mut_rapid_succession() {
        let mut cache = Mru::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, vec![1]);
        cache.insert(2, vec![2]);

        for i in 3..=5 {
            if let Some(mut entry) = cache.peek_mut(&1) {
                entry.push(i);
            }
        }

        assert_eq!(cache.peek(&1), Some(&vec![1, 3, 4, 5]));
        assert_eq!(cache.tail().unwrap().0, &1);
    }

    #[test]
    fn test_mru_peek_mut_alternating_access() {
        let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
        cache.insert("A", 10);
        cache.insert("B", 20);
        cache.insert("C", 30);

        if let Some(mut entry) = cache.peek_mut(&"A") {
            *entry += 1;
        }

        if let Some(mut entry) = cache.peek_mut(&"B") {
            *entry += 2;
        }

        if let Some(mut entry) = cache.peek_mut(&"C") {
            *entry += 3;
        }

        assert_eq!(cache.peek(&"A"), Some(&11));
        assert_eq!(cache.peek(&"B"), Some(&22));
        assert_eq!(cache.peek(&"C"), Some(&33));
        assert_eq!(cache.tail().unwrap().0, &"C");
    }
}

impl_ll_iters!(MruEntry);
