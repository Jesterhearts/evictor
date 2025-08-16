use std::hash::Hash;

use indexmap::IndexMap;

use crate::{
    EntryValue,
    Metadata,
    Policy,
    RandomState,
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
    head: usize,
    tail: usize,
    next_bucket: Option<usize>,
    prev_bucket: Option<usize>,
}

#[derive(Debug, Clone, Default)]
#[doc(hidden)]
pub struct LfuMetadata {
    frequency_head_tail: IndexMap<u64, FreqBucket, RandomState>,
    head_bucket: usize,
}

impl private::Sealed for LfuMetadata {}

impl Metadata for LfuMetadata {
    fn tail_index(&self) -> usize {
        self.frequency_head_tail
            .get_index(self.head_bucket)
            .map_or(0, |(_, bucket)| bucket.tail)
    }
}

impl<T> Policy<T> for LfuPolicy {
    type EntryType = LfuEntry<T>;
    type IntoIter<K> = IntoIter<K, T>;
    type MetadataType = LfuMetadata;

    fn touch_entry<K>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<K, Self::EntryType, RandomState>,
    ) -> usize {
        if index >= queue.len() {
            return index;
        }

        // I.e. if you've been running this cache for hundreds of years (~600 years
        // worth of nanonseconds in a u64), don't do anything to re-order the
        // entry. Honestly, this might even use unreachable.
        if queue[index].frequency == u64::MAX {
            return index;
        }

        let removed = unlink_node(index, metadata, queue);
        debug_assert!(
            removed
                .as_ref()
                .is_none_or(|removed| removed.head == removed.tail && removed.head == index)
        );
        debug_assert!(
            metadata
                .frequency_head_tail
                .get_index(metadata.head_bucket)
                .is_none_or(|(_, bucket)| bucket.prev_bucket.is_none())
        );
        debug_assert!(removed.as_ref().is_none_or(|removed| {
            removed.prev_bucket.is_none_or(|prev| {
                metadata
                    .frequency_head_tail
                    .get_index(prev)
                    .is_some_and(|(freq, _)| *freq < queue[index].frequency)
            }) && removed.next_bucket.is_none_or(|next| {
                metadata
                    .frequency_head_tail
                    .get_index(next)
                    .is_some_and(|(freq, _)| *freq > queue[index].frequency)
            })
        }));

        let old_frequency = queue[index].frequency;
        queue[index].frequency += 1;

        link_node(
            index,
            old_frequency,
            queue[index].frequency,
            removed,
            metadata,
            queue,
        );
        debug_assert!(
            metadata
                .frequency_head_tail
                .get_index(metadata.head_bucket)
                .is_none_or(|(_, bucket)| bucket.prev_bucket.is_none())
        );
        debug_assert!(
            metadata
                .frequency_head_tail
                .get_index(metadata.head_bucket)
                .is_some_and(|(head_freq, bucket)| {
                    bucket.next_bucket.is_none_or(|next| {
                        metadata
                            .frequency_head_tail
                            .get_index(next)
                            .is_some_and(|(freq, _)| *freq > *head_freq)
                    })
                })
        );

        index
    }

    fn swap_remove_entry<K: Hash + Eq>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut IndexMap<K, Self::EntryType, RandomState>,
    ) -> Option<(K, Self::EntryType)> {
        if index >= queue.len() {
            return None;
        }

        unlink_node(index, metadata, queue);

        let result = queue.swap_remove_index(index);
        if index == queue.len() {
            return result;
        }

        if let Some(prev) = queue[index].prev {
            queue[prev].next = Some(index);
        }
        if let Some(next) = queue[index].next {
            queue[next].prev = Some(index);
        }

        if let Some(bucket) = metadata
            .frequency_head_tail
            .get_mut(&queue[index].frequency)
        {
            if bucket.head == queue.len() {
                bucket.head = index;
            }
            if bucket.tail == queue.len() {
                bucket.tail = index;
            }
        }

        result
    }

    fn iter<'q, K>(
        metadata: &'q LfuMetadata,
        queue: &'q IndexMap<K, Self::EntryType, RandomState>,
    ) -> impl Iterator<Item = (&'q K, &'q T)>
    where
        T: 'q,
    {
        let index = metadata
            .frequency_head_tail
            .get_index(metadata.head_bucket)
            .map(|(_, bucket)| bucket.tail);
        Iter {
            queue,
            index,
            freq_bucket: metadata.head_bucket,
            metadata,
        }
    }

    fn into_iter<K>(
        metadata: Self::MetadataType,
        queue: IndexMap<K, Self::EntryType, RandomState>,
    ) -> IntoIter<K, T> {
        let index = metadata
            .frequency_head_tail
            .get_index(metadata.head_bucket)
            .map(|(_, bucket)| bucket.tail);
        IntoIter {
            queue: queue.into_iter().map(Some).collect(),
            index,
            freq_bucket: metadata.head_bucket,
            metadata,
        }
    }

    fn into_entries<K>(
        metadata: Self::MetadataType,
        queue: IndexMap<K, Self::EntryType, RandomState>,
    ) -> impl Iterator<Item = (K, Self::EntryType)> {
        let index = metadata
            .frequency_head_tail
            .get_index(metadata.head_bucket)
            .map(|(_, bucket)| bucket.tail);

        IntoEntriesIter {
            queue: queue.into_iter().map(Some).collect(),
            index,
            freq_bucket: metadata.head_bucket,
            metadata,
        }
    }
}

fn unlink_node<K, T>(
    index: usize,
    metadata: &mut LfuMetadata,
    queue: &mut IndexMap<K, LfuEntry<T>, RandomState>,
) -> Option<FreqBucket> {
    let frequency = queue[index].frequency;
    let prev = queue[index].prev;
    let next = queue[index].next;

    if let Some(prev_idx) = prev {
        queue[prev_idx].next = next;
    }
    if let Some(next_idx) = next {
        queue[next_idx].prev = prev;
    }

    let mut unlinked = None;
    if let Some(bucket) = metadata.frequency_head_tail.get_mut(&frequency) {
        if bucket.head == index {
            if let Some(prev) = prev {
                bucket.head = prev;
            } else if bucket.head == bucket.tail {
                unlinked = Some(unlink_bucket(metadata, frequency));
            }
        } else if bucket.tail == index {
            if let Some(next) = next {
                bucket.tail = next;
            } else if bucket.head == bucket.tail {
                unlinked = Some(unlink_bucket(metadata, frequency));
            }
        }
    }

    queue[index].prev = None;
    queue[index].next = None;

    unlinked
}

fn unlink_bucket(metadata: &mut LfuMetadata, frequency: u64) -> FreqBucket {
    if metadata.frequency_head_tail.len() <= 1 {
        let bucket = metadata
            .frequency_head_tail
            .swap_remove(&frequency)
            .expect("Frequency bucket should exist when unlinking");
        metadata.head_bucket = 0;
        return bucket;
    }

    let (removed_index, _, mut removed) = metadata
        .frequency_head_tail
        .swap_remove_full(&frequency)
        .expect("Frequency bucket should exist");
    let len = metadata.frequency_head_tail.len();

    if removed_index == len {
        if metadata.head_bucket == len {
            metadata.head_bucket = removed.next_bucket.unwrap_or_default();
        }

        let removed_next = removed.next_bucket;
        let removed_prev = removed.prev_bucket;

        if let Some(next) = removed_next {
            metadata.frequency_head_tail[next].prev_bucket = removed_prev;
        }

        if let Some(prev) = removed_prev {
            metadata.frequency_head_tail[prev].next_bucket = removed_next;
        }

        return removed;
    }

    if removed.next_bucket == Some(len) {
        removed.next_bucket = Some(removed_index);
    }

    if removed.prev_bucket == Some(len) {
        removed.prev_bucket = Some(removed_index)
    }

    if metadata.head_bucket == removed_index {
        metadata.head_bucket = removed.next_bucket.unwrap_or_default();
    }
    if metadata.head_bucket == len {
        metadata.head_bucket = removed_index;
    }

    let permuted = &metadata.frequency_head_tail[removed_index];
    let permuted_next = permuted.next_bucket;
    let permuted_prev = permuted.prev_bucket;

    if let Some(next) = permuted_next {
        if next == removed_index {
            metadata.frequency_head_tail[removed_index].next_bucket = None;
        } else {
            metadata.frequency_head_tail[next].prev_bucket = Some(removed_index);
        }
    }

    if let Some(prev) = permuted_prev {
        if prev == removed_index {
            metadata.frequency_head_tail[removed_index].prev_bucket = None;
        } else {
            metadata.frequency_head_tail[prev].next_bucket = Some(removed_index);
        }
    }

    if let Some(prev) = removed.prev_bucket {
        metadata.frequency_head_tail[prev].next_bucket = removed.next_bucket;
    }

    if let Some(next) = removed.next_bucket {
        metadata.frequency_head_tail[next].prev_bucket = removed.prev_bucket;
    }

    removed
}

fn link_node<K, T>(
    node_index: usize,
    prev_frequency: u64,
    frequency: u64,
    removed: Option<FreqBucket>,
    metadata: &mut LfuMetadata,
    queue: &mut IndexMap<K, LfuEntry<T>, RandomState>,
) {
    debug_assert!(prev_frequency < frequency);
    if let Some(bucket) = metadata.frequency_head_tail.get_mut(&frequency) {
        queue[bucket.head].next = Some(node_index);
        queue[node_index].prev = Some(bucket.head);
        queue[node_index].next = None;
        bucket.head = node_index;
        return;
    }

    let insertion_index = metadata.frequency_head_tail.len();

    let is_new_bucket = removed.is_none();
    let mut removed = removed.unwrap_or_default();
    removed.head = node_index;
    removed.tail = node_index;

    if let Some((prev_index, _, bucket)) =
        metadata.frequency_head_tail.get_full_mut(&prev_frequency)
    {
        let old_next = bucket.next_bucket;
        bucket.next_bucket = Some(insertion_index);
        if let Some(next) = old_next {
            metadata.frequency_head_tail[next].prev_bucket = Some(insertion_index);
        }

        removed.next_bucket = old_next;
        removed.prev_bucket = Some(prev_index);

        metadata.frequency_head_tail.insert(frequency, removed);
        return;
    }

    if let Some(prev) = removed.prev_bucket {
        metadata.frequency_head_tail[prev].next_bucket = Some(insertion_index);
    }
    if let Some(next) = removed.next_bucket {
        metadata.frequency_head_tail[next].prev_bucket = Some(insertion_index);
    }
    if metadata.head_bucket != insertion_index
        && (is_new_bucket
            || metadata
                .frequency_head_tail
                .get_index(metadata.head_bucket)
                .map(|kv| *kv.0)
                > Some(frequency))
    {
        metadata.frequency_head_tail[metadata.head_bucket].prev_bucket = Some(insertion_index);
        removed.next_bucket = Some(metadata.head_bucket);
        removed.prev_bucket = None;
        metadata.head_bucket = insertion_index;
    }

    debug_assert!(
        metadata
            .frequency_head_tail
            .get_index(metadata.head_bucket)
            .map(|kv| *kv.0)
            <= Some(frequency)
    );

    metadata.frequency_head_tail.insert(frequency, removed);
}

#[derive(Debug, Clone, Copy)]
struct Iter<'q, K, T> {
    metadata: &'q LfuMetadata,
    queue: &'q IndexMap<K, LfuEntry<T>, RandomState>,
    freq_bucket: usize,
    index: Option<usize>,
}

impl<'q, K, T> Iterator for Iter<'q, K, T> {
    type Item = (&'q K, &'q T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.index {
            let (key, entry) = self.queue.get_index(index)?;
            if let Some(next) = entry.next {
                self.index = Some(next);
            } else {
                let next_bucket = self
                    .metadata
                    .frequency_head_tail
                    .get_index(self.freq_bucket)
                    .and_then(|(_, bucket)| bucket.next_bucket);
                self.freq_bucket = next_bucket.unwrap_or(self.freq_bucket);
                self.index = next_bucket
                    .map(|next_bucket| self.metadata.frequency_head_tail[next_bucket].tail);
            }

            Some((key, entry.value()))
        } else {
            None
        }
    }
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct IntoIter<K, T> {
    metadata: LfuMetadata,
    queue: Vec<Option<(K, LfuEntry<T>)>>,
    freq_bucket: usize,
    index: Option<usize>,
}

impl<K, T> Iterator for IntoIter<K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.index {
            let (key, entry) = self.queue.get_mut(index)?.take()?;
            if let Some(next) = entry.next {
                self.index = Some(next);
            } else {
                let next_bucket = self
                    .metadata
                    .frequency_head_tail
                    .get_index(self.freq_bucket)
                    .and_then(|(_, bucket)| bucket.next_bucket);
                self.freq_bucket = next_bucket.unwrap_or(self.freq_bucket);
                self.index = next_bucket
                    .map(|next_bucket| self.metadata.frequency_head_tail[next_bucket].tail);
            }

            Some((key, entry.into_value()))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct IntoEntriesIter<K, T> {
    metadata: LfuMetadata,
    queue: Vec<Option<(K, LfuEntry<T>)>>,
    freq_bucket: usize,
    index: Option<usize>,
}

impl<K, T> Iterator for IntoEntriesIter<K, T> {
    type Item = (K, LfuEntry<T>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.index {
            let (key, entry) = self.queue.get_mut(index)?.take()?;
            if let Some(next) = entry.next {
                self.index = Some(next);
            } else {
                let next_bucket = self
                    .metadata
                    .frequency_head_tail
                    .get_index(self.freq_bucket)
                    .and_then(|(_, bucket)| bucket.next_bucket);
                self.freq_bucket = next_bucket.unwrap_or(self.freq_bucket);
                self.index = next_bucket
                    .map(|next_bucket| self.metadata.frequency_head_tail[next_bucket].tail);
            }

            Some((key, entry))
        } else {
            None
        }
    }
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
        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(2, "two".to_string())]
        );
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(3, 30), (4, 40), (1, 10)]
        );
    }

    #[test]
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
    fn test_lfu_edge_cases() {
        let mut cache = Lfu::new(NonZeroUsize::new(1).unwrap());

        cache.insert(1, "one");
        assert_eq!(cache.len(), 1);

        cache.insert(2, "two");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.into_iter().collect::<Vec<_>>(), [(2, "two")]);
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [("new1", 4), ("medium", 2), ("high", 3)]
        );
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(4, "four"), (2, "updated_second"), (1, "final_first")]
        );
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(4, 900), (2, 600), (1, 700)]
        );
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

        assert_eq!(
            cache.into_iter().collect::<Vec<_>>(),
            [(2, "two"), (4, "four")]
        );
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

        assert_eq!(cache.into_iter().collect::<Vec<_>>(), [(3, "three")]);
    }

    #[test]
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
    fn test_lfu_retain_none() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        cache.retain(|_, _| false);

        assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
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
    fn test_lfu_iter_empty_cache() {
        let cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());
        let items: Vec<_> = cache.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_lfu_iter_single_item() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one");

        let items: Vec<_> = cache.iter().collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], (&1, &"one"));
    }

    #[test]
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
    fn test_lfu_peek_mut_nonexistent_key() {
        let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one");
        cache.insert(2, "two");

        assert!(cache.peek_mut(&3).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lfu_peek_mut_empty_cache() {
        let mut cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());
        assert!(cache.peek_mut(&1).is_none());
    }

    #[test]
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
