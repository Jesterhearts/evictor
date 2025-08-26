use std::hash::Hash;

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
pub struct LfuPolicy;

impl private::Sealed for LfuPolicy {}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct LfuEntry<T> {
    value: T,
    frequency: u64,
    bucket_ptr: Ptr,
}

impl<T> private::Sealed for LfuEntry<T> {}

impl<T> EntryValue<T> for LfuEntry<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            frequency: 0,
            bucket_ptr: Ptr::null(),
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

#[derive(Debug, Clone, Default)]
struct FreqBucket {
    tail: Ptr,
    head: Ptr,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct LfuMetadata {
    frequency_head_tail: LinkedHashMap<u64, FreqBucket>,
    head_bucket: Ptr,
}

impl Default for LfuMetadata {
    fn default() -> Self {
        let mut frequency_head_tail = LinkedHashMap::default();
        frequency_head_tail.insert_tail(0, FreqBucket::default());
        let head_bucket = frequency_head_tail.head_ptr();
        Self {
            frequency_head_tail,
            head_bucket,
        }
    }
}

impl private::Sealed for LfuMetadata {}

impl<T> Metadata<T> for LfuMetadata {
    type EntryType = LfuEntry<T>;

    #[inline]
    fn candidate_removal_index<K>(&self, queue: &LinkedHashMap<K, LfuEntry<T>>) -> Ptr {
        queue.head_ptr()
    }
}

impl<T> Policy<T> for LfuPolicy {
    type IntoIter<K> = IntoIter<K, T>;
    type MetadataType = LfuMetadata;

    fn insert_or_update_entry<K: Hash + Eq>(
        key: K,
        make_room_on_insert: bool,
        get_value: impl FnOnce(&K, /* is_insert */ bool) -> InsertOrUpdateAction<T>,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) -> crate::InsertionResult<K, T> {
        match queue.entry(key) {
            linked_hashmap::Entry::Occupied(mut occupied_entry) => {
                let ptr = occupied_entry.ptr();
                match get_value(occupied_entry.key(), false) {
                    InsertOrUpdateAction::InsertOrUpdate(value) => {
                        occupied_entry.get_mut().value = value;
                        Self::touch_entry(ptr, metadata, queue);
                        InsertionResult::Updated(ptr)
                    }
                    InsertOrUpdateAction::TouchNoUpdate => {
                        Self::touch_entry(ptr, metadata, queue);
                        InsertionResult::FoundTouchedNoUpdate(ptr)
                    }
                    InsertOrUpdateAction::NoInsert(_) => {
                        unreachable!("Cache hit should not return NoInsert");
                    }
                }
            }
            linked_hashmap::Entry::Vacant(entry) => {
                let value = match get_value(entry.key(), true) {
                    InsertOrUpdateAction::InsertOrUpdate(value) => value,
                    InsertOrUpdateAction::NoInsert(value) => {
                        return InsertionResult::NotFoundNoInsert(entry.into_key(), value);
                    }
                    InsertOrUpdateAction::TouchNoUpdate => {
                        unreachable!("Cache miss should not return TouchNoUpdate");
                    }
                };
                let ptr = if make_room_on_insert {
                    let ptr = entry.push_unlinked(LfuEntry {
                        value,
                        frequency: 0,
                        bucket_ptr: metadata.head_bucket,
                    });
                    Self::evict_entry(metadata, queue);
                    let head_bucket = metadata
                        .frequency_head_tail
                        .ptr_get_mut(metadata.head_bucket)
                        .expect("Head bucket should exist");
                    if head_bucket.head.is_null() {
                        head_bucket.head = ptr;
                    }
                    if head_bucket.tail.is_null() {
                        head_bucket.tail = ptr;
                        queue.link_as_head(ptr);
                    } else {
                        queue.link_node(ptr, head_bucket.tail, Ptr::null(), false);
                        head_bucket.tail = ptr;
                    }
                    ptr
                } else {
                    let head_bucket = metadata
                        .frequency_head_tail
                        .ptr_get_mut(metadata.head_bucket)
                        .expect("Head bucket should exist");
                    let ptr = if head_bucket.tail.is_null() {
                        entry.insert_head(LfuEntry {
                            value,
                            frequency: 0,
                            bucket_ptr: metadata.head_bucket,
                        })
                    } else {
                        entry.insert_after(
                            LfuEntry {
                                value,
                                frequency: 0,
                                bucket_ptr: metadata.head_bucket,
                            },
                            head_bucket.tail,
                        )
                    };
                    head_bucket.tail = ptr;
                    if head_bucket.head.is_null() {
                        head_bucket.head = ptr;
                    }
                    ptr
                };

                InsertionResult::Inserted(ptr)
            }
        }
    }

    fn touch_entry<K: Hash + Eq>(
        index: Ptr,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, LfuEntry<T>>,
    ) {
        // I.e. if you've been running this cache for hundreds of years (~600 years
        // worth of nanonseconds in a u64), don't do anything to re-order the
        // entry. Honestly, this might even use unreachable.
        if queue[index].frequency == u64::MAX {
            return;
        }

        let old_frequency = queue[index].frequency;
        let new_frequency = old_frequency + 1;
        queue[index].frequency = new_frequency;

        let mut bucket_cursor = metadata
            .frequency_head_tail
            .ptr_cursor_mut(queue[index].bucket_ptr);

        let old_prev = queue.prev_ptr(index);
        let old_next = queue.next_ptr(index);

        if let Some((next_freq, next)) = bucket_cursor.next_mut() {
            if *next_freq == new_frequency {
                queue.move_after(index, next.tail);
                next.tail = index;
                debug_assert_ne!(
                    next.head, next.tail,
                    "Head and tail should not be the same after moving an element"
                );
            } else if *next_freq > new_frequency {
                queue.move_before(index, next.head);
                bucket_cursor.insert_after_move_to(
                    new_frequency,
                    FreqBucket {
                        tail: index,
                        head: index,
                    },
                );

                bucket_cursor.move_prev();
            } else {
                queue.move_to_tail(index);
                bucket_cursor.insert_after_move_to(
                    new_frequency,
                    FreqBucket {
                        tail: index,
                        head: index,
                    },
                );

                bucket_cursor.move_prev();
            }
        } else {
            let Some((_, bucket)) = bucket_cursor.current() else {
                unreachable!("Bucket should exist for frequency {old_frequency}");
            };

            queue.move_after(index, bucket.tail);

            bucket_cursor.insert_after_move_to(
                new_frequency,
                FreqBucket {
                    tail: index,
                    head: index,
                },
            );
            bucket_cursor.move_prev();
        }

        queue[index].bucket_ptr = bucket_cursor
            .next_ptr()
            .expect("Just inserted, so next must exist");

        let Some((freq, old)) = bucket_cursor.current_mut() else {
            unreachable!("Bucket should exist for frequency {old_frequency}");
        };

        debug_assert_ne!(
            *freq, new_frequency,
            "Frequency should not be the same after incrementing: {freq} == {new_frequency}"
        );

        debug_assert_eq!(
            *freq, old_frequency,
            "Frequency mismatch: {freq} != {old_frequency}"
        );

        if old.head == old.tail {
            debug_assert_eq!(
                old.head, index,
                "Head and tail should be the same when only one element exists in the bucket: {old:?}",
            );

            if *freq == 0 {
                old.head = Ptr::null();
                old.tail = Ptr::null();
            } else {
                bucket_cursor.remove();
            }
        } else {
            if old.head == index {
                old.head = old_next.unwrap_or_default();
            }
            if old.tail == index {
                old.tail = old_prev.unwrap_or_default();
            }
        }
    }

    fn remove_entry<K: Hash + Eq>(
        ptr: Ptr,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) -> (
        Ptr,
        Option<(K, <Self::MetadataType as Metadata<T>>::EntryType)>,
    ) {
        let Some(removed) = queue.remove_ptr(ptr) else {
            return (Ptr::null(), None);
        };

        finish_removal(ptr, metadata, removed)
    }

    fn remove_key<K: Hash + Eq>(
        key: &K,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) -> Option<<Self::MetadataType as Metadata<T>>::EntryType> {
        let (ptr, removed) = queue.remove(key)?;

        finish_removal(ptr, metadata, removed)
            .1
            .map(|(_, entry)| entry)
    }

    fn iter<'q, K>(
        _: &'q LfuMetadata,
        queue: &'q LinkedHashMap<K, LfuEntry<T>>,
    ) -> impl Iterator<Item = (&'q K, &'q T)>
    where
        T: 'q,
    {
        queue.iter().map(|(k, v)| (k, v.value()))
    }

    fn into_iter<K>(_: Self::MetadataType, queue: LinkedHashMap<K, LfuEntry<T>>) -> IntoIter<K, T> {
        IntoIter {
            inner: queue.into_iter(),
        }
    }

    fn into_entries<K>(
        _: Self::MetadataType,
        queue: LinkedHashMap<K, LfuEntry<T>>,
    ) -> impl Iterator<Item = (K, LfuEntry<T>)> {
        queue.into_iter()
    }

    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn debug_validate<K: std::fmt::Debug>(
        metadata: &Self::MetadataType,
        queue: &LinkedHashMap<K, LfuEntry<T>>,
    ) where
        T: std::fmt::Debug,
    {
        queue.debug_validate();
        if metadata.frequency_head_tail.is_empty() {
            assert!(
                queue.is_empty(),
                "Queue should be empty when no frequency buckets exist: {metadata:#?} {queue:#?}",
            );
            return;
        }

        let mut prev_frequency = None;

        for (freq, bucket) in metadata.frequency_head_tail.iter() {
            if let Some(prev_freq) = prev_frequency {
                assert!(
                    *freq > prev_freq,
                    "Frequency buckets are not in increasing order: {prev_freq} >= {freq}, {metadata:#?}, {queue:#?}",
                );
            }
            prev_frequency = Some(*freq);
            assert!(
                queue.contains_ptr(bucket.head),
                "Bucket head index out of bounds: {bucket:?}, {metadata:#?}, {queue:#?}",
            );
            assert!(
                queue.contains_ptr(bucket.tail),
                "Bucket tail index out of bounds: {bucket:?}, {metadata:#?}, {queue:#?}",
            );
        }
    }
}

fn finish_removal<K, T>(
    ptr: Ptr,
    metadata: &mut LfuMetadata,
    removed: RemovedEntry<K, LfuEntry<T>>,
) -> (Ptr, Option<(K, LfuEntry<T>)>) {
    let mut bucket_cursor = metadata
        .frequency_head_tail
        .ptr_cursor_mut(removed.value.bucket_ptr);

    let Some((_, bucket)) = bucket_cursor.current_mut() else {
        unreachable!(
            "Bucket should exist for frequency {}",
            removed.value.frequency
        );
    };

    if bucket.head == bucket.tail {
        debug_assert_eq!(
            bucket.head, ptr,
            "Head and tail should be the same when only one element exists in the bucket"
        );

        if removed.value.bucket_ptr == metadata.head_bucket {
            bucket.head = Ptr::null();
            bucket.tail = Ptr::null();
        } else {
            bucket_cursor.remove();
        }
    } else {
        if bucket.head == ptr {
            bucket.head = removed.next;
        }
        if bucket.tail == ptr {
            bucket.tail = removed.prev;
        }
    }

    (removed.next, Some((removed.key, removed.value)))
}

#[doc(hidden)]
pub struct IntoIter<K, T> {
    inner: linked_hashmap::IntoIter<K, LfuEntry<T>>,
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

    use crate::Lfu;

    #[test]
    #[timeout(5000)]
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
        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (1, "one".to_string()),
                (2, "two".to_string()),
                (3, "three".to_string())
            ]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_cache_eviction_policy() {
        let mut cache = Lfu::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.get(&1);
        cache.get(&1);

        cache.insert(3, "three".to_string());

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(3, "three".to_string()), (1, "one".to_string())]
        );
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_lfu_cache_clear() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_cache_extend() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());

        let items = vec![(2, "two".to_string()), (3, "three".to_string())];
        cache.extend(items);

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (1, "one".to_string()),
                (2, "two".to_string()),
                (3, "three".to_string())
            ]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_cache_from_iterator() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string()),
        ];

        let cache: Lfu<i32, String> = items.into_iter().collect();

        assert_eq!(cache.capacity(), 3);
        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (1, "one".to_string()),
                (2, "two".to_string()),
                (3, "three".to_string())
            ]
        )
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_cache_from_iter_overlapping() {
        let items = vec![
            (1, "one".to_string()),
            (3, "three".to_string()),
            (3, "three_new".to_string()),
            (2, "two".to_string()),
        ];

        let cache: Lfu<i32, String> = items.into_iter().collect();

        assert_eq!(cache.capacity(), 3);
        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (1, "one".to_string()),
                (2, "two".to_string()),
                (3, "three_new".to_string())
            ]
        )
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_edge_case_capacity_one() {
        let mut cache = Lfu::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one".to_string());
        assert_eq!(cache.len(), 1);

        cache.insert(2, "two".to_string());
        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(2, "two".to_string())]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_frequency_ordering_consistency() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());
        assert_eq!(
            cache.iter().collect::<Vec<_>>(),
            [
                (&1, &"one".to_string()),
                (&2, &"two".to_string()),
                (&3, &"three".to_string())
            ]
        );

        cache.get(&1);
        cache.get(&1);
        cache.get(&2);

        let least_frequent = cache.tail().map(|(k, _)| *k);
        assert_eq!(least_frequent, Some(3));

        cache.get(&3);
        cache.get(&3);
        cache.get(&3);

        let least_frequent: Option<i32> = cache.tail().map(|(k, _)| *k);
        assert_eq!(least_frequent, Some(2));
    }

    #[test]
    #[timeout(5000)]
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (4, "four".to_string()),
                (2, "two".to_string()),
                (1, "one".to_string())
            ]
        );
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (6, "value_6".to_string()),
                (7, "value_7".to_string()),
                (1, "value_1".to_string()),
                (2, "value_2".to_string()),
                (3, "value_3".to_string()),
            ]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_cache_get_or_insert_with_frequency_update() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");

        let value = cache.get_or_insert_with(1, |_| "new_one");
        assert_eq!(*value, "one");

        let value = cache.get_or_insert_with(3, |_| "three");
        assert_eq!(*value, "three");

        cache.insert(4, "four");

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(3, "three"), (4, "four"), (1, "one")]
        );
    }

    #[test]
    #[timeout(5000)]
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
        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (3, "three"),
                (4, "four"),
                (1, "one_updated"),
                (2, "two_updated"),
            ]
        )
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_cache_insert_mut_frequency_behavior() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        let val1 = cache.insert_mut(1, "one".to_string());
        val1.push_str("_modified");

        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let val1_again = cache.get_mut(&1).unwrap();
        val1_again.push_str("_again");

        cache.insert(4, "four".to_string());

        assert!(cache.contains_key(&1));
        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (3, "three".to_string()),
                (4, "four".to_string()),
                (1, "one_modified_again".to_string())
            ]
        )
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_lfu_capacity_constraints() {
        let cache = Lfu::<i32, i32>::new(NonZeroUsize::new(5).unwrap());
        assert_eq!(cache.capacity(), 5);

        let cache = Lfu::<i32, i32>::new(NonZeroUsize::new(1).unwrap());
        assert_eq!(cache.capacity(), 1);

        let cache = Lfu::<i32, i32>::new(NonZeroUsize::new(100).unwrap());
        assert_eq!(cache.capacity(), 100);
    }

    #[test]
    #[timeout(5000)]
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (5, "five"),
                (6, "six"),
                (3, "three"),
                (2, "two"),
                (1, "one"),
            ]
        )
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_tie_breaking_behavior() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.insert(4, "four");

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&4));

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(2, "two"), (3, "three"), (4, "four"),]
        );
    }

    #[test]
    #[timeout(5000)]
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
        let mut current_state: Vec<_> = cache.into_iter().collect();
        current_state.sort();
        assert_eq!(current_state, [("A", 1), ("B", 2), ("C", 3), ("E", 5)]);
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(3, 30), (4, 40), (1, 10)]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_mutable_references() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        let val = cache.insert_mut(1, "test".to_string());
        val.push_str("_modified");

        cache.insert(2, "second".to_string());
        cache.insert(3, "third".to_string());

        if let Some(val) = cache.get_mut(&1) {
            val.push_str("_again");
        }

        cache.insert(4, "fourth".to_string());

        assert!(cache.contains_key(&1));
        assert_eq!(cache.peek(&1), Some(&"test_modified_again".to_string()));
        assert!(cache.contains_key(&4));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_edge_cases() {
        let mut cache = Lfu::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one");
        assert_eq!(cache.len(), 1);

        cache.insert(2, "two");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.into_iter().collect::<Vec<_>>(), [(2, "two")]);
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_lfu_consistent_eviction_policy() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert("low", 1);
        cache.insert("medium", 2);
        cache.insert("high", 3);

        cache.get(&"medium");
        cache.get(&"high");
        cache.get(&"high");

        cache.insert("new1", 4);

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [("new1", 4), ("medium", 2), ("high", 3)]
        );
    }

    #[test]
    #[timeout(5000)]
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(4, "four"), (2, "updated_second"), (1, "final_first")]
        );
    }

    #[test]
    #[timeout(5000)]
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(4, 900), (2, 600), (1, 700)]
        );
    }

    #[test]
    #[timeout(5000)]
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (4, "four"),
                (5, "five"),
                (2, "updated_two"),
                (3, "final_three")
            ]
        );
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_lfu_retain_basic() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");
        cache.insert(5, "five");

        cache.retain(|&k, _| k % 2 == 0);

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(2, "two"), (4, "four")]
        );
    }

    #[test]
    #[timeout(5000)]
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

        assert_eq!(cache.into_iter().collect::<Vec<_>>(), [(3, "three")]);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_retain_all() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.retain(|_, _| true);

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(1, "one"), (2, "two"), (3, "three")]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_retain_none() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.retain(|_, _| false);

        assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_retain_empty_cache() {
        let mut cache = Lfu::<i32, &str>::new(NonZeroUsize::new(3).unwrap());

        cache.retain(|_, _| true);

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_lfu_retain_with_mutable_values() {
        let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());

        cache.insert(1, "short".to_string());
        cache.insert(2, "medium".to_string());
        cache.insert(3, "very_long_string".to_string());
        cache.insert(4, "tiny".to_string());
        cache.insert(5, "another_long_string".to_string());

        cache.retain(|_, v| v.len() > 5);

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [
                (2, "medium".to_string()),
                (3, "very_long_string".to_string()),
                (5, "another_long_string".to_string())
            ]
        );
    }

    #[test]
    #[timeout(5000)]
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(5, "five"), (3, "three"), (4, "four")]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_empty_cache() {
        let cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());
        let items: Vec<_> = cache.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_single_item() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one");

        let items: Vec<_> = cache.iter().collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], (&1, &"one"));
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_eviction_order() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(1, "one"), (2, "two"), (3, "three")]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_with_different_frequencies() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.get(&1);
        cache.get(&2);
        cache.get(&2);

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(3, "three"), (1, "one"), (2, "two")]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_complex_frequency_pattern() {
        let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");

        cache.get(&1);
        cache.get(&1);
        cache.get(&1);

        cache.get(&2);

        cache.get(&3);
        cache.get(&3);

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(4, "four"), (2, "two"), (3, "three"), (1, "one")]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_same_frequency_order() {
        let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");

        cache.get(&1);
        cache.get(&2);
        cache.get(&3);
        cache.get(&4);

        let items: Vec<_> = cache.iter().collect();
        assert_eq!(
            items,
            [(&1, &"one"), (&2, &"two"), (&3, &"three"), (&4, &"four")]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_after_eviction() {
        let mut cache = Lfu::new(NonZeroUsize::new(2).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");

        cache.get(&2);

        cache.insert(3, "three");

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(3, "three"), (2, "two")]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_consistency_with_tail() {
        let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());

        cache.insert(10, "ten");
        cache.insert(20, "twenty");
        cache.insert(30, "thirty");
        cache.insert(40, "forty");

        cache.get(&20);
        cache.get(&30);
        cache.get(&30);

        let tail = cache.tail();
        let mut iter_items = cache.iter();
        let first_iter_item = iter_items.next();

        assert_eq!(tail, first_iter_item);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_multiple_iterations() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.get(&1);
        cache.get(&1);

        let items1: Vec<_> = cache.iter().collect();
        let items2: Vec<_> = cache.iter().collect();

        assert_eq!(items1, items2);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_after_remove() {
        let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");
        cache.insert(4, "four");

        cache.get(&2);
        cache.get(&2);
        cache.get(&3);

        cache.remove(&3);

        let items: Vec<_> = cache.iter().collect();
        assert_eq!(items, [(&1, &"one"), (&4, &"four"), (&2, &"two")]);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_frequency_bucket_ordering() {
        let mut cache = Lfu::new(NonZeroUsize::new(6).unwrap());

        for i in 1..=6 {
            cache.insert(i, format!("value_{}", i));
        }

        for _ in 0..3 {
            cache.get(&1);
            cache.get(&2);
        }

        for _ in 0..2 {
            cache.get(&3);
            cache.get(&4);
        }

        cache.get(&5);
        cache.get(&6);

        let items: Vec<_> = cache.iter().collect();
        assert_eq!(
            items,
            [
                (&5, &"value_5".to_string()),
                (&6, &"value_6".to_string()),
                (&3, &"value_3".to_string()),
                (&4, &"value_4".to_string()),
                (&1, &"value_1".to_string()),
                (&2, &"value_2".to_string())
            ]
        );
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_iter_with_updates() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.get(&1);

        *cache.get_mut(&1).unwrap() = "updated_one";

        let items: Vec<_> = cache.iter().collect();
        assert_eq!(items, [(&2, &"two"), (&3, &"three"), (&1, &"updated_one")]);
    }

    #[test]
    #[timeout(5000)]
    fn into_iter_matches_iter() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        let items: Vec<_> = cache.iter().map(|(k, v)| (*k, *v)).collect();
        let into_items: Vec<_> = cache.into_iter().collect();

        assert_eq!(items, into_items);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_peek_mut_no_modification() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
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
    #[timeout(5000)]
    fn test_lfu_peek_mut_with_modification() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
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
    #[timeout(5000)]
    fn test_lfu_peek_mut_nonexistent_key() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one");
        cache.insert(2, "two");

        assert!(cache.peek_mut(&3).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_peek_mut_empty_cache() {
        let mut cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());
        assert!(cache.peek_mut(&1).is_none());
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_peek_mut_single_item() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());

        if let Some(mut entry) = cache.peek_mut(&1) {
            entry.push_str("_modified");
        }

        assert_eq!(cache.peek(&1), Some(&"one_modified".to_string()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_peek_mut_frequency_updates() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);

        cache.get(&1);
        cache.get(&1);
        cache.get(&2);

        assert_eq!(cache.tail().unwrap().0, &3);

        if let Some(mut entry) = cache.peek_mut(&3) {
            *entry += 100;
        }

        assert_eq!(cache.peek(&3), Some(&130));
        assert_eq!(cache.tail().unwrap().0, &2);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_peek_mut_with_eviction() {
        let mut cache = Lfu::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, 10);
        cache.insert(2, 20);

        if let Some(mut entry) = cache.peek_mut(&1) {
            *entry += 5;
        }

        cache.insert(3, 30);

        assert_eq!(cache.peek(&1), Some(&15));
        assert!(cache.peek(&2).is_none());
        assert_eq!(cache.peek(&3), Some(&30));
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_peek_mut_preserve_frequency_on_no_modification() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);

        cache.get(&1);
        cache.get(&1);
        cache.get(&2);

        let original_tail = cache.tail().map(|(k, _)| *k);

        if let Some(entry) = cache.peek_mut(&3) {
            let _value = *entry;
        }

        let new_tail = cache.tail().map(|(k, _)| *k);
        assert_eq!(original_tail, new_tail);
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_peek_mut_multiple_modifications() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
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
    #[timeout(5000)]
    fn test_lfu_peek_mut_ordering_consistency() {
        let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);
        cache.insert(4, 400);

        cache.get(&1);
        cache.get(&1);
        cache.get(&1);
        cache.get(&3);
        cache.get(&3);
        cache.get(&3);
        cache.get(&4);
        cache.get(&4);

        let expected: Vec<_> = cache
            .iter()
            .map(|(k, v)| (*k, if *k != 2 { *v } else { *v + 1000 }))
            .collect();

        if let Some(mut entry) = cache.peek_mut(&2) {
            *entry += 1000;
        }

        let after_modification: Vec<_> = cache.iter().map(|(k, v)| (*k, *v)).collect();

        assert_eq!(expected, after_modification);

        assert_eq!(cache.peek(&2), Some(&1200));
    }

    #[test]
    #[timeout(5000)]
    fn test_lfu_peek_mut_complex_scenario() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
        cache.insert("low", 1);
        cache.insert("medium", 2);
        cache.insert("high", 3);

        cache.get(&"medium");
        cache.get(&"high");
        cache.get(&"high");

        assert_eq!(cache.tail().unwrap().0, &"low");

        if let Some(mut entry) = cache.peek_mut(&"low") {
            *entry += 100;
        }

        assert_eq!(cache.peek(&"low"), Some(&101));
        assert_eq!(cache.tail().unwrap().0, &"medium");

        cache.insert("new", 4);

        assert!(cache.contains_key(&"low"));
        assert!(!cache.contains_key(&"medium"));
        assert!(cache.contains_key(&"high"));
        assert!(cache.contains_key(&"new"));
    }
}
