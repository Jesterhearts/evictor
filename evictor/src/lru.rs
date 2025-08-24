use crate::{
    EntryValue,
    InsertOrUpdateAction,
    InsertionResult,
    Metadata,
    Policy,
    linked_hashmap::{
        self,
        LinkedHashMap,
        Ptr,
        RemovedEntry,
    },
    private,
};

#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct LruPolicy;

impl private::Sealed for LruPolicy {}

#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct LruEntry<T> {
    value: T,
}

impl<T> private::Sealed for LruEntry<T> {}

impl<T> EntryValue<T> for LruEntry<T> {
    fn new(value: T) -> Self {
        Self { value }
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

#[derive(Debug, Clone, Copy, Default)]
#[doc(hidden)]
pub struct LruMetadata;

impl private::Sealed for LruMetadata {}

impl<T> Metadata<T> for LruMetadata {
    type EntryType = LruEntry<T>;

    fn candidate_removal_index<K>(&self, queue: &LinkedHashMap<K, LruEntry<T>>) -> Ptr {
        queue.head_ptr()
    }
}

impl<T> Policy<T> for LruPolicy {
    type IntoIter<K> = IntoIter<K, T>;
    type MetadataType = LruMetadata;

    fn insert_or_update_entry<K: std::hash::Hash + Eq>(
        key: K,
        make_room_on_insert: bool,
        get_value: impl FnOnce(&K, /* is_insert */ bool) -> InsertOrUpdateAction<T>,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) -> InsertionResult<K, T> {
        match queue.entry(key) {
            linked_hashmap::Entry::Occupied(mut occupied_entry) => {
                let ptr = occupied_entry.ptr();
                match get_value(occupied_entry.key(), false) {
                    InsertOrUpdateAction::TouchNoUpdate => {
                        Self::touch_entry(ptr, metadata, queue);
                        InsertionResult::FoundTouchedNoUpdate(ptr)
                    }
                    InsertOrUpdateAction::InsertOrUpdate(value) => {
                        *occupied_entry.get_mut() = LruEntry::new(value);
                        Self::touch_entry(ptr, metadata, queue);
                        InsertionResult::Updated(ptr)
                    }
                    InsertOrUpdateAction::NoInsert(_) => {
                        unreachable!("Cache hit should not return NoInsert");
                    }
                }
            }
            linked_hashmap::Entry::Vacant(vacant_entry) => {
                match get_value(vacant_entry.key(), true) {
                    InsertOrUpdateAction::NoInsert(value) => {
                        InsertionResult::NotFoundNoInsert(vacant_entry.into_key(), value)
                    }
                    InsertOrUpdateAction::TouchNoUpdate => {
                        unreachable!("Cache miss should not return TouchNoUpdate");
                    }
                    InsertOrUpdateAction::InsertOrUpdate(value) => {
                        vacant_entry.insert_tail(LruEntry::new(value));
                        if make_room_on_insert {
                            Self::evict_entry(metadata, queue);
                        }
                        InsertionResult::Inserted(queue.tail_ptr())
                    }
                }
            }
        }
    }

    fn touch_entry<K: std::hash::Hash + Eq>(
        ptr: Ptr,
        _: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) {
        queue.move_to_tail(ptr);
    }

    fn remove_entry<K: std::hash::Hash + Eq>(
        ptr: Ptr,
        _: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) -> (
        Ptr,
        Option<(K, <Self::MetadataType as Metadata<T>>::EntryType)>,
    ) {
        let Some(RemovedEntry {
            key, value, next, ..
        }) = queue.swap_remove_ptr(ptr)
        else {
            return (Ptr::null(), None);
        };
        (next, Some((key, value)))
    }

    fn iter<'q, K>(
        _: &'q Self::MetadataType,
        queue: &'q LinkedHashMap<K, LruEntry<T>>,
    ) -> impl Iterator<Item = (&'q K, &'q T)>
    where
        T: 'q,
    {
        queue.iter().map(|(k, v)| (k, v.value()))
    }

    fn into_iter<K>(_: Self::MetadataType, queue: LinkedHashMap<K, LruEntry<T>>) -> IntoIter<K, T> {
        IntoIter {
            inner: queue.into_iter(),
        }
    }

    fn into_entries<K>(
        _: Self::MetadataType,
        queue: LinkedHashMap<K, LruEntry<T>>,
    ) -> impl Iterator<Item = (K, LruEntry<T>)> {
        queue.into_iter()
    }

    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn debug_validate<K: std::hash::Hash + Eq + std::fmt::Debug>(
        _: &Self::MetadataType,
        queue: &LinkedHashMap<K, LruEntry<T>>,
    ) where
        T: std::fmt::Debug,
    {
        queue.debug_validate();
    }
}

#[doc(hidden)]
pub struct IntoIter<K, T> {
    inner: linked_hashmap::IntoIter<K, LruEntry<T>>,
}

impl<K, T> Iterator for IntoIter<K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k, v.into_value()))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use ntest::timeout;

    use crate::Lru;

    #[test]
    #[timeout(1000)]
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
    #[timeout(1000)]
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
    #[timeout(1000)]
    fn test_lru_eviction_order() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        lru.insert(4, 40);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(2, 20), (3, 30), (4, 40)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_access_updates_order() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        lru.get(&1);

        lru.insert(4, 40);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(3, 30), (1, 10), (4, 40)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_multiple_accesses() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);

        lru.get(&1);
        lru.get(&1);
        lru.get(&1);

        lru.insert(3, 30);

        assert_eq!(lru.into_iter().collect::<Vec<_>>(), [(1, 10), (3, 30)]);
    }

    #[test]
    #[timeout(1000)]
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
    #[timeout(1000)]
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
    #[timeout(1000)]
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

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(2, 20), (1, 10), (3, 30), (5, 50)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_interleaved_operations() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.get(&1);
        lru.insert(3, 30);
        lru.get(&2);
        lru.insert(4, 40);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(3, 30), (2, 20), (4, 40)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_get_or_insert_with_eviction() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);

        lru.get(&1);

        let value = lru.get_or_insert_with(3, |_| 30);
        assert_eq!(value, &30);

        assert_eq!(lru.into_iter().collect::<Vec<_>>(), [(1, 10), (3, 30)]);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_get_or_insert_existing_key() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);
        lru.insert(3, 30);

        let value = lru.get_or_insert_with(1, |_| 999);
        assert_eq!(value, &10);

        lru.insert(4, 40);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(3, 30), (1, 10), (4, 40)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_sequential_pattern() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        for i in 1..=10 {
            lru.insert(i, i * 10);
        }

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(8, 80), (9, 90), (10, 100)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_repeated_access_single_key() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        lru.insert(1, 10);
        lru.insert(2, 20);

        for _ in 0..100 {
            lru.get(&1);
        }

        lru.insert(3, 30);

        assert_eq!(lru.into_iter().collect::<Vec<_>>(), [(1, 10), (3, 30)]);
    }

    #[test]
    #[timeout(1000)]
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
    #[timeout(1000)]
    fn test_lru_capacity_constraints() {
        let lru = Lru::<i32, i32>::new(NonZeroUsize::new(5).unwrap());
        assert_eq!(lru.capacity(), 5);

        let lru = Lru::<i32, i32>::new(NonZeroUsize::new(1).unwrap());
        assert_eq!(lru.capacity(), 1);

        let lru = Lru::<i32, i32>::new(NonZeroUsize::new(100).unwrap());
        assert_eq!(lru.capacity(), 100);
    }

    #[test]
    #[timeout(1000)]
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
    #[timeout(1000)]
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

        assert_eq!(lru.into_iter().collect::<Vec<_>>(), [(2, 20), (3, 30)]);
    }

    #[test]
    #[timeout(1000)]
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

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(1, 10), (3, 30), (4, 40)]
        );
    }

    #[test]
    #[timeout(1000)]
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

        // Test empty cache ordering
        assert_eq!(lru.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    #[timeout(1000)]
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
    #[timeout(1000)]
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
        assert!(lru.tail().is_none());

        assert_eq!(lru.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_mutable_access() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());
        lru.insert(1, String::from("hello"));
        lru.insert(2, String::from("world"));

        if let Some(val) = lru.get_mut(&1) {
            val.push_str(" modified");
        }

        assert_eq!(lru.get(&1), Some(&String::from("hello modified")));

        lru.insert(3, String::from("new"));

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [
                (1, String::from("hello modified")),
                (3, String::from("new"))
            ]
        );
    }

    #[test]
    #[timeout(1000)]
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
    #[timeout(1000)]
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
    #[timeout(1000)]
    fn test_lru_extend() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());
        lru.insert(1, 10);

        let items = vec![(2, 20), (3, 30), (4, 40)];
        lru.extend(items);

        assert_eq!(lru.len(), 4);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(1, 10), (2, 20), (3, 30), (4, 40)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_from_iterator() {
        let items = vec![(1, 10), (2, 20), (3, 30)];
        let lru: Lru<i32, i32> = items.into_iter().collect();

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.capacity(), 3);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(1, 10), (2, 20), (3, 30)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_from_iterator_with_eviction() {
        let items = vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)];
        let lru: Lru<i32, i32> = items.into_iter().collect();

        assert_eq!(lru.len(), 5);
        assert_eq!(lru.capacity(), 5);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_shrink_to_fit() {
        let mut lru = Lru::new(NonZeroUsize::new(10).unwrap());
        lru.insert(1, 10);
        lru.insert(2, 20);

        lru.shrink_to_fit();

        assert_eq!(lru.len(), 2);

        assert_eq!(lru.into_iter().collect::<Vec<_>>(), [(1, 10), (2, 20)]);
    }

    #[test]
    #[timeout(1000)]
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
        lru.insert(6, "sixth");

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(1, "first"), (2, "second"), (5, "fifth"), (6, "sixth")]
        );
    }

    #[test]
    #[timeout(1000)]
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

        // Verify that recent items are still present
        for i in 15..20 {
            assert!(lru.contains_key(&i));
        }
    }

    #[test]
    #[timeout(1000)]
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
    #[timeout(1000)]
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
    #[timeout(1000)]
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
    #[timeout(1000)]
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
    #[timeout(1000)]
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

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [
                (2, "two"),
                (3, "three"),
                (4, "four"),
                (5, "five"),
                (1, "updated_one"),
                (6, "six")
            ]
        );
    }

    #[test]
    #[timeout(1000)]
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
    #[timeout(1000)]
    fn test_lru_retain_basic() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");
        lru.insert(4, "four");
        lru.insert(5, "five");

        lru.retain(|&k, _| k % 2 == 0);

        assert_eq!(lru.len(), 2);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(2, "two"), (4, "four")]
        );
    }

    #[test]
    #[timeout(1000)]
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

        assert_eq!(lru.into_iter().collect::<Vec<_>>(), [(3, "three")]);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_retain_all() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        lru.retain(|_, _| true);

        assert_eq!(lru.len(), 3);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(1, "one"), (2, "two"), (3, "three")]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_retain_none() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        lru.retain(|_, _| false);

        assert_eq!(lru.len(), 0);
        assert!(lru.is_empty());

        assert_eq!(lru.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_retain_empty_cache() {
        let mut lru = Lru::<i32, &str>::new(NonZeroUsize::new(3).unwrap());

        lru.retain(|_, _| true);

        assert_eq!(lru.len(), 0);
        assert!(lru.is_empty());
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_retain_single_item() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");

        lru.retain(|&k, _| k == 1);

        assert_eq!(lru.len(), 1);
        assert!(lru.contains_key(&1));

        lru.retain(|&k, _| k != 1);

        assert_eq!(lru.len(), 0);
        assert_eq!(lru.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    #[timeout(1000)]
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

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [
                (3, "three"),
                (4, "four"),
                (5, "five"),
                (6, "six"),
                (7, "seven")
            ]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_retain_with_mutable_values() {
        let mut lru = Lru::new(NonZeroUsize::new(5).unwrap());

        lru.insert(1, String::from("short"));
        lru.insert(2, String::from("medium"));
        lru.insert(3, String::from("very_long_string"));
        lru.insert(4, String::from("tiny"));
        lru.insert(5, String::from("another_long_string"));

        lru.retain(|_, v| v.len() > 5);

        assert_eq!(lru.len(), 3);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [
                (2, String::from("medium")),
                (3, String::from("very_long_string")),
                (5, String::from("another_long_string"))
            ]
        );
    }

    #[test]
    #[timeout(1000)]
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

        let expected_keys: Vec<_> = (0..100)
            .filter(|&x| x % 3 == 0)
            .map(|x| (x, x * 10))
            .collect();
        assert_eq!(lru.into_iter().collect::<Vec<_>>(), expected_keys);
    }

    #[test]
    #[timeout(1000)]
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

        lru.insert(6, "six");
        lru.get(&3);
        assert_eq!(lru.len(), 4);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(5, "five"), (4, "four"), (6, "six"), (3, "three")]
        );
    }

    #[test]
    #[timeout(1000)]
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

        lru.insert(11, "new_item".to_string());
        assert_eq!(lru.len(), 6);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [
                (2, "value_2".to_string()),
                (4, "value_4".to_string()),
                (6, "value_6".to_string()),
                (8, "value_8".to_string()),
                (10, "value_10".to_string()),
                (11, "new_item".to_string())
            ]
        );
    }

    #[test]
    #[timeout(1000)]
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
        assert_eq!(lru.tail(), Some((&2, &"two")));

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(2, "two"), (4, "four"), (1, "one")]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_retain_order_preservation() {
        let mut lru = Lru::new(NonZeroUsize::new(6).unwrap());

        for i in 1..=6 {
            lru.insert(i, i * 10);
        }

        lru.get(&2);
        lru.get(&4);
        lru.get(&6);

        let before_retain: Vec<_> = lru
            .iter()
            .filter_map(|(k, v)| (k % 2 == 0).then_some((*k, *v)))
            .collect();

        lru.retain(|&k, _| k % 2 == 0);

        assert_eq!(lru.len(), 3);
        assert_eq!(lru.tail(), Some((&2, &20)));
        assert_eq!(
            lru.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            before_retain,
        );

        lru.insert(8, 80);
        lru.insert(10, 100);
        lru.insert(12, 120);
        lru.insert(14, 140);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(4, 40), (6, 60), (8, 80), (10, 100), (12, 120), (14, 140)]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_empty_cache() {
        let lru = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
        let items: Vec<_> = lru.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_single_item() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());
        lru.insert(1, "one");

        let items: Vec<_> = lru.iter().collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], (&1, &"one"));
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_eviction_order() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(1, "one"), (2, "two"), (3, "three")]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_after_access() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        lru.get(&1);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(2, "two"), (3, "three"), (1, "one")]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_complex_access_pattern() {
        let mut lru = Lru::new(NonZeroUsize::new(4).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");
        lru.insert(4, "four");

        lru.get(&2);
        lru.get(&4);
        lru.get(&1);

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(3, "three"), (2, "two"), (4, "four"), (1, "one")]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_after_update() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        *lru.get_mut(&1).unwrap() = "updated_one";

        let items: Vec<_> = lru.iter().collect();
        assert_eq!(items, [(&2, &"two"), (&3, &"three"), (&1, &"updated_one")]);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_with_eviction() {
        let mut lru = Lru::new(NonZeroUsize::new(2).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        assert_eq!(
            lru.into_iter().collect::<Vec<_>>(),
            [(2, "two"), (3, "three")]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_consistency_with_tail() {
        let mut lru = Lru::new(NonZeroUsize::new(4).unwrap());

        lru.insert(10, "ten");
        lru.insert(20, "twenty");
        lru.insert(30, "thirty");
        lru.insert(40, "forty");

        lru.get(&20);
        lru.get(&30);

        let tail = lru.tail();
        let mut iter_items = lru.iter();
        let first_iter_item = iter_items.next();

        assert_eq!(tail, first_iter_item);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_multiple_iterations() {
        let mut lru = Lru::new(NonZeroUsize::new(3).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");

        let items1: Vec<_> = lru.iter().collect();
        let items2: Vec<_> = lru.iter().collect();

        assert_eq!(items1, items2);
        assert_eq!(items1.len(), 3);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_iter_after_remove() {
        let mut lru = Lru::new(NonZeroUsize::new(4).unwrap());

        lru.insert(1, "one");
        lru.insert(2, "two");
        lru.insert(3, "three");
        lru.insert(4, "four");

        lru.get(&2);
        lru.remove(&3);

        let items: Vec<_> = lru.iter().collect();
        assert_eq!(items, [(&1, &"one"), (&4, &"four"), (&2, &"two")]);
    }

    #[test]
    #[timeout(1000)]
    fn into_iter_matches_iter() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        let items: Vec<_> = cache.iter().map(|(k, v)| (*k, *v)).collect();
        let into_items: Vec<_> = cache.into_iter().collect();

        assert_eq!(items, into_items);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_no_modification() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
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
    #[timeout(1000)]
    fn test_lru_peek_mut_with_modification() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
        cache.insert("A", vec![1, 2, 3]);
        cache.insert("B", vec![4, 5, 6]);
        cache.insert("C", vec![7, 8, 9]);

        assert_eq!(cache.tail().unwrap().0, &"A");

        if let Some(mut entry) = cache.peek_mut(&"A") {
            entry.push(4);
        }

        assert_eq!(cache.peek(&"A"), Some(&vec![1, 2, 3, 4]));
        assert_eq!(cache.tail().unwrap().0, &"B");
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_nonexistent_key() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one");
        cache.insert(2, "two");

        assert!(cache.peek_mut(&3).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_empty_cache() {
        let mut cache = Lru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
        assert!(cache.peek_mut(&1).is_none());
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_single_item() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());

        if let Some(mut entry) = cache.peek_mut(&1) {
            entry.push_str("_modified");
        }

        assert_eq!(cache.peek(&1), Some(&"one_modified".to_string()));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.tail().unwrap().0, &1);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_with_eviction() {
        let mut cache = Lru::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, 10);
        cache.insert(2, 20);

        assert_eq!(cache.tail().unwrap().0, &1);

        if let Some(mut entry) = cache.peek_mut(&1) {
            *entry += 5;
        }

        assert_eq!(cache.tail().unwrap().0, &2);

        cache.insert(3, 30);

        assert_eq!(cache.peek(&1), Some(&15));
        assert!(cache.peek(&2).is_none());
        assert_eq!(cache.peek(&3), Some(&30));
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_preserve_order_on_no_modification() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
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
    #[timeout(1000)]
    fn test_lru_peek_mut_multiple_modifications() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "hello".to_string());
        cache.insert(2, "world".to_string());

        if let Some(mut entry) = cache.peek_mut(&1) {
            entry.push_str(" there");
        }

        if let Some(mut entry) = cache.peek_mut(&1) {
            entry.push_str(" friend");
        }

        assert_eq!(cache.peek(&1), Some(&"hello there friend".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_ordering_consistency() {
        let mut cache = Lru::new(NonZeroUsize::new(4).unwrap());
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
        assert_eq!(after_modification[0].0, 4);
        assert_eq!(cache.peek(&3), Some(&1300));
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_complex_scenario() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
        cache.insert("first", 1);
        cache.insert("second", 2);
        cache.insert("third", 3);

        cache.get(&"first");

        assert_eq!(cache.tail().unwrap().0, &"second");

        if let Some(mut entry) = cache.peek_mut(&"second") {
            *entry += 100;
        }

        assert_eq!(cache.peek(&"second"), Some(&102));
        assert_eq!(cache.tail().unwrap().0, &"third");

        cache.insert("fourth", 4);

        assert!(cache.contains_key(&"second"));
        assert!(cache.contains_key(&"first"));
        assert!(!cache.contains_key(&"third"));
        assert!(cache.contains_key(&"fourth"));
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_rapid_succession() {
        let mut cache = Lru::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, vec![1]);
        cache.insert(2, vec![2]);

        for i in 3..=5 {
            if let Some(mut entry) = cache.peek_mut(&1) {
                entry.push(i);
            }
        }

        assert_eq!(cache.peek(&1), Some(&vec![1, 3, 4, 5]));
        assert_eq!(cache.tail().unwrap().0, &2);
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_alternating_access() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
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
        assert_eq!(cache.tail().unwrap().0, &"A");
    }

    #[test]
    #[timeout(1000)]
    fn test_lru_peek_mut_interaction_with_get() {
        let mut cache = Lru::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);

        assert_eq!(cache.tail().unwrap().0, &1);

        if let Some(mut entry) = cache.peek_mut(&1) {
            *entry += 100;
        }

        assert_eq!(cache.tail().unwrap().0, &2);

        cache.get(&2);

        assert_eq!(cache.tail().unwrap().0, &3);

        if let Some(mut entry) = cache.peek_mut(&3) {
            *entry += 200;
        }

        assert_eq!(cache.tail().unwrap().0, &1);
        assert_eq!(cache.peek(&1), Some(&110));
        assert_eq!(cache.peek(&3), Some(&230));
    }
}
