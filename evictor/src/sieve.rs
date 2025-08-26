use std::{
    collections::VecDeque,
    hash::Hash,
};

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
pub struct SievePolicy;

impl private::Sealed for SievePolicy {}

#[derive(Debug, Clone, Copy, Default)]
pub struct SieveMetadata {
    hand: Ptr,
}

impl private::Sealed for SieveMetadata {}

impl<Value> Metadata<Value> for SieveMetadata {
    type EntryType = SieveEntry<Value>;

    fn candidate_removal_index<K>(&self, queue: &LinkedHashMap<K, SieveEntry<Value>>) -> Ptr {
        if queue.is_empty() {
            return Ptr::default();
        }

        let mut visited = vec![true; queue.len()];
        let mut hand = self.hand;
        while queue[hand].visited && visited[queue.ptr_index_unstable(hand).unwrap_or_default()] {
            visited[queue.ptr_index_unstable(hand).unwrap_or_default()] = false;
            hand = queue.prev_ptr(hand).unwrap_or(queue.tail_ptr());
        }

        hand
    }
}

#[derive(Debug, Clone)]
pub struct SieveEntry<V> {
    value: V,
    visited: bool,
}

impl<Value> private::Sealed for SieveEntry<Value> {}

impl<Value> EntryValue<Value> for SieveEntry<Value> {
    fn new(value: Value) -> Self {
        SieveEntry {
            value,
            visited: false,
        }
    }

    fn value(&self) -> &Value {
        &self.value
    }

    fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }

    fn into_value(self) -> Value {
        self.value
    }
}

impl<Value> Policy<Value> for SievePolicy {
    type IntoIter<K> = IntoIter<K, Value>;
    type MetadataType = SieveMetadata;

    fn insert_or_update_entry<K: Hash + Eq>(
        key: K,
        make_room_on_insert: bool,
        get_value: impl FnOnce(&K, /* is_insert */ bool) -> InsertOrUpdateAction<Value>,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<Value>>::EntryType>,
    ) -> InsertionResult<K, Value> {
        match queue.entry(key) {
            linked_hashmap::Entry::Occupied(mut occupied_entry) => {
                let ptr = occupied_entry.ptr();
                match get_value(occupied_entry.key(), false) {
                    InsertOrUpdateAction::TouchNoUpdate => {
                        let entry = occupied_entry.get_mut();
                        entry.visited = true;
                        InsertionResult::FoundTouchedNoUpdate(ptr)
                    }
                    InsertOrUpdateAction::NoInsert(_) => {
                        unreachable!("Cache hit should not return NoInsert");
                    }
                    InsertOrUpdateAction::InsertOrUpdate(value) => {
                        let entry = occupied_entry.get_mut();
                        entry.value = value;
                        entry.visited = true;
                        InsertionResult::Updated(ptr)
                    }
                }
            }
            linked_hashmap::Entry::Vacant(vacant_entry) => {
                let value = match get_value(vacant_entry.key(), true) {
                    InsertOrUpdateAction::TouchNoUpdate => {
                        unreachable!("Cache miss should not return TouchNoUpdate")
                    }
                    InsertOrUpdateAction::NoInsert(value) => {
                        return InsertionResult::NotFoundNoInsert(vacant_entry.into_key(), value);
                    }
                    InsertOrUpdateAction::InsertOrUpdate(value) => value,
                };
                let ptr = if make_room_on_insert {
                    let ptr = vacant_entry.push_unlinked(SieveEntry::new(value));
                    Self::evict_entry(metadata, queue);
                    queue.link_as_head(ptr);
                    ptr
                } else {
                    vacant_entry.insert_head(SieveEntry::new(value))
                };
                if metadata.hand == Ptr::default() {
                    metadata.hand = ptr;
                }
                InsertionResult::Inserted(ptr)
            }
        }
    }

    fn touch_entry<K: std::hash::Hash + Eq>(
        ptr: Ptr,
        _: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<Value>>::EntryType>,
    ) {
        queue[ptr].visited = true;
    }

    fn evict_entry<K: Hash + Eq>(
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, SieveEntry<Value>>,
    ) -> Option<(K, SieveEntry<Value>)> {
        if queue.is_empty() {
            return None;
        }

        move_hand_to_eviction_index(metadata, queue);
        Self::remove_entry(metadata.hand, metadata, queue).1
    }

    fn remove_entry<K: Hash + Eq>(
        ptr: Ptr,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<Value>>::EntryType>,
    ) -> (
        Ptr,
        Option<(K, <Self::MetadataType as Metadata<Value>>::EntryType)>,
    ) {
        let Some(RemovedEntry {
            key,
            value,
            prev,
            next,
            ..
        }) = queue.remove_ptr(ptr)
        else {
            return (Ptr::null(), None);
        };
        if ptr == metadata.hand {
            metadata.hand = prev;
        }
        (next, Some((key, value)))
    }

    fn remove_key<K: Hash + Eq>(
        key: &K,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<Value>>::EntryType>,
    ) -> Option<<Self::MetadataType as Metadata<Value>>::EntryType> {
        let (ptr, removed) = queue.remove(key)?;
        if ptr == metadata.hand {
            metadata.hand = removed.prev;
        }
        Some(removed.value)
    }

    fn iter<'q, K>(
        metadata: &'q Self::MetadataType,
        queue: &'q LinkedHashMap<K, SieveEntry<Value>>,
    ) -> impl Iterator<Item = (&'q K, &'q Value)>
    where
        Value: 'q,
    {
        if queue.is_empty() {
            return Iter {
                queue,
                skipped: VecDeque::new(),
                index: Ptr::default(),
                seen_count: 0,
            };
        }

        let mut skipped = VecDeque::new();
        let mut index = metadata.hand;

        let mut entry = queue.ptr_get(index).expect("Hand pointer invalid");
        while entry.visited {
            skipped.push_back(index);

            if skipped.len() == queue.len() {
                return Iter {
                    queue,
                    seen_count: skipped.len(),
                    skipped,
                    index: Ptr::default(),
                };
            }

            index = queue.prev_ptr(index).unwrap_or(queue.tail_ptr());
            entry = queue.ptr_get(index).unwrap();
        }

        Iter {
            index,
            queue,
            seen_count: skipped.len(),
            skipped,
        }
    }

    fn into_iter<K>(
        metadata: Self::MetadataType,
        queue: LinkedHashMap<K, SieveEntry<Value>>,
    ) -> Self::IntoIter<K> {
        IntoIter {
            inner: build_into_entries_iter(metadata, queue),
        }
    }

    fn into_entries<K>(
        metadata: Self::MetadataType,
        queue: LinkedHashMap<K, SieveEntry<Value>>,
    ) -> impl Iterator<Item = (K, SieveEntry<Value>)> {
        build_into_entries_iter(metadata, queue)
    }

    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn debug_validate<K: std::hash::Hash + Eq + std::fmt::Debug>(
        _: &Self::MetadataType,
        queue: &LinkedHashMap<K, SieveEntry<Value>>,
    ) where
        Value: std::fmt::Debug,
    {
        queue.debug_validate();
    }
}

fn move_hand_to_eviction_index<K, Value>(
    metadata: &mut SieveMetadata,
    queue: &mut LinkedHashMap<K, SieveEntry<Value>>,
) {
    while queue[metadata.hand].visited {
        queue[metadata.hand].visited = false;
        metadata.hand = queue.prev_ptr(metadata.hand).unwrap_or(queue.tail_ptr());
    }
    debug_assert!(!queue[metadata.hand].visited);
}

fn build_into_entries_iter<K, Value>(
    metadata: SieveMetadata,
    mut queue: LinkedHashMap<K, SieveEntry<Value>>,
) -> IntoEntriesIter<K, Value> {
    let mut skipped: VecDeque<(K, SieveEntry<Value>)> = VecDeque::new();

    if queue.ptr_get(metadata.hand).is_none_or(|e| !e.visited) {
        return IntoEntriesIter {
            skipped,
            queue,
            index: metadata.hand,
        };
    }

    let mut entry = queue
        .remove_ptr(metadata.hand)
        .expect("Hand pointer invalid");
    let mut index = entry.prev.or(queue.tail_ptr());
    loop {
        skipped.push_back((entry.key, entry.value));

        let Some(e) = queue.remove_ptr(index) else {
            break;
        };
        entry = e;
        index = entry.prev.or(queue.tail_ptr());
    }

    IntoEntriesIter {
        skipped,
        index,
        queue,
    }
}

#[derive(Debug, Clone)]
struct Iter<'q, K, T> {
    queue: &'q LinkedHashMap<K, SieveEntry<T>>,
    skipped: VecDeque<Ptr>,
    seen_count: usize,
    index: Ptr,
}

impl<'q, K, T> Iterator for Iter<'q, K, T> {
    type Item = (&'q K, &'q T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.seen_count != self.queue.len()
            && let Some((mut key, mut entry)) = self.queue.ptr_get_entry(self.index)
        {
            while entry.visited {
                self.seen_count += 1;
                self.skipped.push_back(self.index);
                if self.seen_count == self.queue.len() {
                    self.index = Ptr::default();
                    break;
                }
                self.index = self
                    .queue
                    .prev_ptr(self.index)
                    .unwrap_or(self.queue.tail_ptr());
                let (k, e) = self.queue.ptr_get_entry(self.index).unwrap();
                key = k;
                entry = e;
            }

            if !entry.visited {
                self.seen_count += 1;
                self.index = self
                    .queue
                    .prev_ptr(self.index)
                    .unwrap_or(self.queue.tail_ptr());
                return Some((key, &entry.value));
            }
        }

        self.skipped.pop_front().and_then(|ptr| {
            let (key, entry) = self.queue.ptr_get_entry(ptr)?;
            Some((key, &entry.value))
        })
    }
}

#[derive(Debug)]
#[doc(hidden)]
struct IntoEntriesIter<K, T> {
    skipped: VecDeque<(K, SieveEntry<T>)>,
    queue: LinkedHashMap<K, SieveEntry<T>>,
    index: Ptr,
}

impl<K, V> Iterator for IntoEntriesIter<K, V> {
    type Item = (K, SieveEntry<V>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(mut entry) = self.queue.remove_ptr(self.index) {
            self.index = entry.prev.or(self.queue.tail_ptr());

            if !entry.value.visited {
                return Some((entry.key, entry.value));
            }

            loop {
                self.skipped.push_back((entry.key, entry.value));
                let Some(e) = self.queue.remove_ptr(self.index) else {
                    break;
                };
                self.index = e.prev.or(self.queue.tail_ptr());
                if !e.value.visited {
                    return Some((e.key, e.value));
                }
                entry = e;
            }
        }

        self.skipped.pop_front()
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct IntoIter<K, T> {
    inner: IntoEntriesIter<K, T>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, e)| (k, e.value))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        num::NonZeroUsize,
    };

    use ntest::timeout;

    use crate::Sieve;

    #[test]
    #[timeout(5000)]
    fn test_sieve_basic_operations() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 3);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());
        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));

        let tail = cache.tail();
        let first = cache.iter().next();
        assert_eq!(tail, first);
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_eviction_respects_recent_access() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.get(&1);
        cache.insert(3, "three".to_string());
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&3));
        assert!(!cache.contains_key(&2));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_peek_does_not_affect_eviction() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.get(&2);
        cache.peek(&1);
        cache.insert(3, "three".to_string());
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(!cache.contains_key(&1));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_pop() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(10, "a".to_string());
        cache.insert(20, "b".to_string());

        let popped = cache.pop().unwrap();
        assert_eq!(popped.0, 10);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&20));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_retain() {
        let mut cache = Sieve::new(NonZeroUsize::new(5).unwrap());
        for i in 0..5 {
            cache.insert(i, format!("v{i}"));
        }

        cache.retain(|k, _| k % 2 == 0);
        let keys: Vec<_> = cache.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, [0, 2, 4]);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_from_iterator_overlapping_keys() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (1, "one_new".to_string()),
            (3, "three".to_string()),
            (2, "two_new".to_string()),
        ];
        let cache: Sieve<i32, String> = items.into_iter().collect();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert_eq!(cache.peek(&1), Some(&"one_new".to_string()));
        assert_eq!(cache.peek(&2), Some(&"two_new".to_string()));
        assert_eq!(cache.peek(&3), Some(&"three".to_string()));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_capacity_one() {
        let mut cache = Sieve::new(NonZeroUsize::new(1).unwrap());
        cache.insert(1, 10);
        assert_eq!(cache.len(), 1);
        cache.insert(2, 20);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&2));
        cache.get(&2);
        cache.insert(3, 30);
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_empty_cache() {
        let mut cache: Sieve<i32, String> = Sieve::new(NonZeroUsize::new(5).unwrap());
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 5);
        assert_eq!(cache.peek(&1), None);
        assert_eq!(cache.get(&1), None);
        assert!(!cache.contains_key(&1));
        assert_eq!(cache.pop(), None);
        assert_eq!(cache.iter().count(), 0);
        assert_eq!(cache.keys().count(), 0);
        assert_eq!(cache.values().count(), 0);
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_get_mut() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        if let Some(value) = cache.get_mut(&1) {
            value.push_str("_modified");
        }
        assert_eq!(cache.peek(&1), Some(&"one_modified".to_string()));

        cache.insert(3, "three".to_string());
        cache.insert(4, "four".to_string());
        assert!(cache.contains_key(&1));
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_remove() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let removed = cache.remove(&2);
        assert_eq!(removed, Some("two".to_string()));
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&3));

        assert_eq!(cache.remove(&5), None);

        cache.remove(&1);
        cache.remove(&3);
        assert!(cache.is_empty());
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_clear() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.len(), 3);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 3);

        cache.insert(4, "four".to_string());
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&4));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_into_iter() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        let items: Vec<_> = cache.into_iter().collect();
        assert_eq!(items.len(), 3);

        let keys: Vec<_> = items.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_visited_bit_behavior() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        cache.get(&1);
        cache.get(&3);

        cache.insert(4, "four".to_string());
        assert!(!cache.contains_key(&2));
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));

        cache.insert(5, "five".to_string());
        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&4) || cache.contains_key(&5));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_hand_pointer_movement() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        cache.get(&1);
        cache.get(&2);

        cache.insert(3, "three".to_string());
        assert_eq!(cache.len(), 2);

        let has_1 = cache.contains_key(&1);
        let has_2 = cache.contains_key(&2);
        assert!(has_1 ^ has_2);
        assert!(cache.contains_key(&3));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_extend() {
        let mut cache = Sieve::new(NonZeroUsize::new(5).unwrap());
        cache.insert(1, "one".to_string());

        let items = vec![
            (2, "two".to_string()),
            (3, "three".to_string()),
            (1, "one_updated".to_string()),
        ];

        cache.extend(items);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.peek(&1), Some(&"one_updated".to_string()));
        assert_eq!(cache.peek(&2), Some(&"two".to_string()));
        assert_eq!(cache.peek(&3), Some(&"three".to_string()));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_retain_complex() {
        let mut cache = Sieve::new(NonZeroUsize::new(6).unwrap());
        for i in 1..=6 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.get(&2);
        cache.get(&4);
        cache.get(&6);

        cache.retain(|k, _| *k % 2 == 0);

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&4));
        assert!(cache.contains_key(&6));
        assert!(!cache.contains_key(&1));
        assert!(!cache.contains_key(&3));
        assert!(!cache.contains_key(&5));
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_stress_eviction_patterns() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());

        for round in 0..10 {
            let base = round * 3;
            cache.insert(base, format!("val_{}", base));
            cache.insert(base + 1, format!("val_{}", base + 1));
            cache.insert(base + 2, format!("val_{}", base + 2));

            cache.get(&base);

            if round < 9 {
                let next_base = (round + 1) * 3;
                cache.insert(next_base, format!("val_{}", next_base));

                if cache.contains_key(&base) {}
            }
        }

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_clone() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.get(&1);

        let cloned = cache.clone();
        assert_eq!(cache.len(), cloned.len());
        assert_eq!(cache.capacity(), cloned.capacity());
        assert_eq!(cache.peek(&1), cloned.peek(&1));
        assert_eq!(cache.peek(&2), cloned.peek(&2));

        cache.insert(3, "three".to_string());
        assert_ne!(cache.len(), cloned.len());
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_debug_format() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Sieve"));

        assert!(debug_str.len() > 10);
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_from_iter_capacity_exceeded() {
        let items: Vec<(i32, String)> = (0..10).map(|i| (i, format!("value_{}", i))).collect();

        let cache: Sieve<i32, String> = items.into_iter().collect();
        assert_eq!(cache.capacity(), 10);
        assert_eq!(cache.len(), 10);

        for i in 0..10 {
            assert!(cache.contains_key(&i));
        }
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_iterator_stability_during_modification() {
        let mut cache = Sieve::new(NonZeroUsize::new(5).unwrap());
        for i in 0..5 {
            cache.insert(i, format!("val_{}", i));
        }

        let items_before: HashSet<_> = cache.iter().map(|(k, v)| (*k, v.clone())).collect();

        cache.get(&1);
        cache.get(&3);

        let items_after: HashSet<_> = cache.iter().map(|(k, v)| (*k, v.clone())).collect();
        assert_eq!(items_before, items_after);
    }

    #[test]
    #[timeout(5000)]
    fn test_sieve_pop_until_empty() {
        let mut cache = Sieve::new(NonZeroUsize::new(4).unwrap());
        cache.insert('a', 1);
        cache.insert('b', 2);
        cache.insert('c', 3);
        cache.insert('d', 4);

        cache.get(&'b');
        cache.get(&'d');

        let mut popped = Vec::new();
        while let Some((k, v)) = cache.pop() {
            popped.push((k, v));
        }

        assert_eq!(popped.len(), 4);
        assert!(cache.is_empty());

        let keys: Vec<_> = popped.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&'a'));
        assert!(keys.contains(&'b'));
        assert!(keys.contains(&'c'));
        assert!(keys.contains(&'d'));
    }

    #[test]
    #[timeout(5000)]
    fn iterator_next_is_tail_is_pop() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.iter().next(), cache.tail());
        let tail = cache.tail().map(|(k, v)| (*k, v.clone()));
        let popped = cache.pop();
        assert_eq!(tail, popped);
    }

    #[test]
    #[timeout(5000)]
    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn fuzz_1() {
        let mut cache = Sieve::new(NonZeroUsize::new(1).unwrap());
        cache.insert(255, 0);
        cache.insert(0, 0);
        cache.debug_validate();
    }

    #[test]
    #[timeout(5000)]
    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn fuzz_2() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(0, 5);
        cache.get_or_insert_with(255, |_| 43);
        cache.get_or_insert_with(0, |_| 0);
        cache.debug_validate();
    }

    #[test]
    #[timeout(5000)]
    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn fuzz_3() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(40, 247);
        cache.get_or_insert_with(7, |_| 255);
        cache.retain(|k, _| k % 2 == 0);
        cache.debug_validate();
    }

    #[test]
    #[timeout(5000)]
    fn fuzz_4() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(0, 255);
        cache.insert(159, 159);
        cache.insert(0, 10);
        cache.retain(|k, _| k % 2 == 0);
        assert!(cache.contains_key(&0));
        assert_eq!(cache.into_iter().collect::<Vec<_>>(), [(0, 10)]);
    }

    #[test]
    #[timeout(5000)]
    fn fuzz_5() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.insert(1, 1);
        cache.insert(3, 3);
        cache.insert(5, 5);
        cache.retain(|k, _| k % 2 == 0);
        cache.insert(0, 10);
        assert_eq!(cache.into_iter().collect::<Vec<_>>(), [(0, 10)]);
    }

    #[test]
    #[timeout(5000)]
    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn fuzz_6() {
        let mut cache = Sieve::new(NonZeroUsize::new(3).unwrap());
        cache.get(&10);
        cache.insert(1, 0);
        cache.peek(&248);
        cache.insert(181, 181);
        cache.get(&181);
        cache.remove(&1);
        assert_eq!(cache.iter().collect::<Vec<_>>(), [(&181, &181)]);
        cache.debug_validate();
    }

    #[test]
    #[timeout(5000)]
    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn fuzz_7() {
        let mut cache = Sieve::new(NonZeroUsize::new(2).unwrap());
        cache.insert(0, 0);
        cache.insert(1, 0);
        cache.insert(2, 0);

        cache.debug_validate();
    }

    #[test]
    #[timeout(5000)]
    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn fuzz_8() {
        let mut cache = Sieve::new(NonZeroUsize::new(103).unwrap());
        cache.get_or_insert_with(213, |_| 6);
        cache.get_or_insert_with(255, |_| 1);
        cache.get_or_insert_with(213, |_| 6);
        let will_pop = cache.iter().next().map(|(k, v)| (*k, *v));
        let popped = cache.pop();
        assert_eq!(will_pop, popped);
        cache.debug_validate();
    }

    #[test]
    #[timeout(5000)]
    fn fuzz_9() {
        let mut cache = Sieve::new(NonZeroUsize::new(33).unwrap());
        cache.get_or_insert_with(2, |_| 0);
        cache.insert(189, 10);
        cache.insert(207, 47);
        cache.retain(|k, _| k % 2 == 0);
        assert_eq!(cache.into_iter().collect::<Vec<_>>(), [(2, 0)]);
    }

    #[test]
    #[timeout(5000)]
    fn fuzz_10() {
        let mut cache = Sieve::new(NonZeroUsize::new(33).unwrap());
        cache.get_or_insert_with(2, |_| 113);
        cache.insert(47, 8);
        cache.insert(47, 113);
        assert_eq!(cache.iter().collect::<Vec<_>>(), [(&2, &113), (&47, &113)]);
        cache.retain(|k, _| k % 2 == 0);
    }
}
