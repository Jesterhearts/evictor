use std::collections::HashSet;

use crate::{
    EntryValue,
    Metadata,
    Policy,
    private,
    utils::swap_remove_ll_entry,
};

#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct SIEVEPolicy;

impl private::Sealed for SIEVEPolicy {}

#[derive(Debug, Clone, Copy, Default)]
pub struct SIEVEMetadata {
    head: usize,
    tail: usize,
    hand: usize,
}

impl private::Sealed for SIEVEMetadata {}

impl Metadata for SIEVEMetadata {
    fn candidate_removal_index(&self) -> usize {
        self.hand
    }
}

#[derive(Debug, Clone)]
pub struct SIEVEEntry<V> {
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
    visited: bool,
}

impl<Value> private::Sealed for SIEVEEntry<Value> {}

impl<Value> EntryValue<Value> for SIEVEEntry<Value> {
    fn new(value: Value) -> Self {
        SIEVEEntry {
            value,
            prev: None,
            next: None,
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

impl<Value> Policy<Value> for SIEVEPolicy {
    type EntryType = SIEVEEntry<Value>;
    type IntoIter<K> = IntoIter<K, Value>;
    type MetadataType = SIEVEMetadata;

    fn touch_entry<K>(
        mut index: usize,
        make_room: bool,
        metadata: &mut Self::MetadataType,
        queue: &mut indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> usize {
        if index >= queue.len() {
            return index;
        }

        if queue.len() == 1 {
            return index;
        }

        if make_room {
            while queue[metadata.hand].visited {
                queue[metadata.hand].visited = false;
                metadata.hand = queue[metadata.hand].prev.unwrap_or(metadata.tail);
            }
            debug_assert!(!queue[metadata.hand].visited);
            let victim = metadata.hand;
            let next_hand = queue[victim].prev.unwrap_or(metadata.tail);
            if index == queue.len() - 1 {
                // We are going to swap remove the item @ hand, so we need to update the index
                // to point at what will become its new position.
                index = victim;
            }
            Self::swap_remove_entry(victim, metadata, queue);
            metadata.hand = next_hand;
        }

        let is_new_entry = queue[index].prev.is_none() && queue[index].next.is_none();
        if is_new_entry {
            queue[metadata.head].prev = Some(index);
            queue[index].next = Some(metadata.head);
            metadata.head = index;
        }

        queue[index].visited = !is_new_entry;

        index
    }

    fn swap_remove_entry<K>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> Option<(K, Self::EntryType)> {
        if index >= queue.len() {
            return None;
        }

        metadata.hand = queue[metadata.hand].prev.unwrap_or(metadata.tail);
        if metadata.hand == queue.len().saturating_sub(1) {
            metadata.hand = index;
        }
        swap_remove_ll_entry!(index, metadata, queue)
    }

    fn iter<'q, K>(
        metadata: &'q Self::MetadataType,
        queue: &'q indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> impl Iterator<Item = (&'q K, &'q Value)>
    where
        Value: 'q,
    {
        Iter {
            index: (!queue.is_empty()).then_some(metadata.candidate_removal_index()),
            metadata,
            queue,
            seen: HashSet::with_capacity_and_hasher(queue.len(), crate::RandomState::default()),
            wrapped: false,
        }
    }

    fn into_iter<K>(
        metadata: Self::MetadataType,
        queue: indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> Self::IntoIter<K> {
        IntoIter {
            index: (!queue.is_empty()).then_some(metadata.candidate_removal_index()),
            metadata,
            seen: HashSet::with_capacity_and_hasher(queue.len(), crate::RandomState::default()),
            queue: queue.into_iter().map(EntryOrPrev::from).collect(),
            wrapped: false,
        }
    }

    fn into_entries<K>(
        metadata: Self::MetadataType,
        queue: indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> impl Iterator<Item = (K, Self::EntryType)> {
        IntoEntriesIter {
            index: (!queue.is_empty()).then_some(metadata.candidate_removal_index()),
            metadata,
            seen: HashSet::with_capacity_and_hasher(queue.len(), crate::RandomState::default()),
            queue: queue.into_iter().map(EntryOrPrev::from).collect(),
            wrapped: false,
        }
    }

    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn debug_validate<K: std::hash::Hash + Eq>(
        metadata: &Self::MetadataType,
        queue: &indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) {
        use crate::utils::validate_ll;

        validate_ll!(metadata, queue);
    }
}

#[derive(Debug, Clone)]
struct Iter<'q, K, T> {
    metadata: &'q SIEVEMetadata,
    queue: &'q indexmap::IndexMap<K, SIEVEEntry<T>, crate::RandomState>,
    seen: HashSet<usize, crate::RandomState>,
    wrapped: bool,
    index: Option<usize>,
}

impl<'q, K, T> Iterator for Iter<'q, K, T> {
    type Item = (&'q K, &'q T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(mut index) = self.index {
            let entry = &self.queue.get_index(index)?;

            loop {
                let prev = self.queue[index].prev;
                let prev = if let Some(prev) = prev {
                    prev
                } else if !self.wrapped {
                    self.wrapped = true;
                    self.metadata.tail
                } else {
                    self.index = None;
                    break;
                };

                if (!self.seen.contains(&prev) && self.queue[prev].visited)
                    || (self.seen.contains(&prev) && !self.queue[prev].visited)
                {
                    self.seen.insert(index);
                    index = prev;
                } else {
                    self.seen.insert(index);
                    self.index = Some(prev);
                    break;
                }
            }

            Some((entry.0, entry.1.value()))
        } else {
            None
        }
    }
}

#[derive(Debug)]
enum EntryOrPrev<K, T> {
    Entry((K, SIEVEEntry<T>)),
    Prev(Option<usize>),
}

impl<K, T> EntryOrPrev<K, T> {
    fn prev(&self) -> Option<usize> {
        match self {
            EntryOrPrev::Entry((_, entry)) => entry.prev,
            EntryOrPrev::Prev(prev) => *prev,
        }
    }

    fn visited(&self) -> bool {
        match self {
            EntryOrPrev::Entry((_, entry)) => entry.visited,
            EntryOrPrev::Prev(_) => true,
        }
    }
}

impl<K, T> Default for EntryOrPrev<K, T> {
    fn default() -> Self {
        EntryOrPrev::Prev(None)
    }
}

impl<K, T> From<(K, SIEVEEntry<T>)> for EntryOrPrev<K, T> {
    fn from(entry: (K, SIEVEEntry<T>)) -> Self {
        EntryOrPrev::Entry(entry)
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct IntoIter<K, T> {
    metadata: SIEVEMetadata,
    queue: Vec<EntryOrPrev<K, T>>,
    seen: HashSet<usize, crate::RandomState>,
    wrapped: bool,
    index: Option<usize>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(mut index) = self.index {
            let EntryOrPrev::Entry(entry) = std::mem::take(&mut self.queue[index]) else {
                return None;
            };

            self.queue[index] = EntryOrPrev::Prev(entry.1.prev);

            loop {
                let prev = self.queue[index].prev();
                let prev = if let Some(prev) = prev {
                    prev
                } else if !self.wrapped {
                    self.wrapped = true;
                    self.metadata.tail
                } else {
                    self.index = None;
                    break;
                };

                if (!self.seen.contains(&prev) && self.queue[prev].visited())
                    || (self.seen.contains(&prev) && !self.queue[prev].visited())
                {
                    self.seen.insert(index);
                    index = prev;
                } else {
                    self.seen.insert(index);
                    self.index = Some(prev);
                    break;
                }
            }

            Some((entry.0, entry.1.into_value()))
        } else {
            None
        }
    }
}

struct IntoEntriesIter<K, T> {
    metadata: SIEVEMetadata,
    queue: Vec<EntryOrPrev<K, T>>,
    seen: HashSet<usize, crate::RandomState>,
    wrapped: bool,
    index: Option<usize>,
}

impl<K, V> Iterator for IntoEntriesIter<K, V> {
    type Item = (K, SIEVEEntry<V>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(mut index) = self.index {
            let EntryOrPrev::Entry(entry) = std::mem::take(&mut self.queue[index]) else {
                debug_assert!(false, "Expected EntryOrPrev::Entry at index {}", index);
                return None;
            };
            self.queue[index] = EntryOrPrev::Prev(entry.1.prev);

            loop {
                let prev = self.queue[index].prev();
                let prev = if let Some(prev) = prev {
                    prev
                } else if !self.wrapped {
                    self.wrapped = true;
                    self.metadata.tail
                } else {
                    self.index = None;
                    break;
                };

                if (!self.seen.contains(&prev) && self.queue[prev].visited())
                    || (self.seen.contains(&prev) && !self.queue[prev].visited())
                {
                    self.seen.insert(index);
                    index = prev;
                } else {
                    self.seen.insert(index);
                    self.index = Some(prev);
                    break;
                }
            }

            Some(entry)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        num::NonZeroUsize,
    };

    use crate::SIEVE;

    #[test]
    fn test_sieve_basic_operations() {
        let mut cache = SIEVE::new(NonZeroUsize::new(3).unwrap());
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
    fn test_sieve_eviction_respects_recent_access() {
        let mut cache = SIEVE::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.get(&1);
        cache.insert(3, "three".to_string());
        assert!(cache.contains_key(&1));
        assert!(cache.contains_key(&3));
        assert!(!cache.contains_key(&2));
    }

    #[test]
    fn test_sieve_peek_does_not_affect_eviction() {
        let mut cache = SIEVE::new(NonZeroUsize::new(2).unwrap());
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
    fn test_sieve_pop() {
        let mut cache = SIEVE::new(NonZeroUsize::new(2).unwrap());
        cache.insert(10, "a".to_string());
        cache.insert(20, "b".to_string());

        let popped = cache.pop().unwrap();
        assert_eq!(popped.0, 10);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_key(&20));
    }

    #[test]
    fn test_sieve_retain() {
        let mut cache = SIEVE::new(NonZeroUsize::new(5).unwrap());
        for i in 0..5 {
            cache.insert(i, format!("v{i}"));
        }

        cache.retain(|k, _| k % 2 == 0);
        let keys: Vec<_> = cache.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, [0, 2, 4]);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_sieve_from_iterator_overlapping_keys() {
        let items = vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (1, "one_new".to_string()),
            (3, "three".to_string()),
            (2, "two_new".to_string()),
        ];
        let cache: SIEVE<i32, String> = items.into_iter().collect();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert_eq!(cache.peek(&1), Some(&"one_new".to_string()));
        assert_eq!(cache.peek(&2), Some(&"two_new".to_string()));
        assert_eq!(cache.peek(&3), Some(&"three".to_string()));
    }

    #[test]
    fn test_sieve_capacity_one() {
        let mut cache = SIEVE::new(NonZeroUsize::new(1).unwrap());
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
    fn test_sieve_empty_cache() {
        let mut cache: SIEVE<i32, String> = SIEVE::new(NonZeroUsize::new(5).unwrap());
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
    fn test_sieve_get_mut() {
        let mut cache = SIEVE::new(NonZeroUsize::new(3).unwrap());
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
    fn test_sieve_remove() {
        let mut cache = SIEVE::new(NonZeroUsize::new(3).unwrap());
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
    fn test_sieve_clear() {
        let mut cache = SIEVE::new(NonZeroUsize::new(3).unwrap());
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
    fn test_sieve_into_iter() {
        let mut cache = SIEVE::new(NonZeroUsize::new(3).unwrap());
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
    fn test_sieve_visited_bit_behavior() {
        let mut cache = SIEVE::new(NonZeroUsize::new(3).unwrap());
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
    fn test_sieve_hand_pointer_movement() {
        let mut cache = SIEVE::new(NonZeroUsize::new(2).unwrap());
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
    fn test_sieve_large_capacity() {
        let mut cache = SIEVE::new(NonZeroUsize::new(1000).unwrap());

        for i in 0..500 {
            cache.insert(i, format!("value_{}", i));
        }
        assert_eq!(cache.len(), 500);

        for i in (0..500).step_by(10) {
            cache.get(&i);
        }

        for i in 500..1000 {
            cache.insert(i, format!("value_{}", i));
        }
        assert_eq!(cache.len(), 1000);

        cache.insert(1000, "value_1000".to_string());
        assert_eq!(cache.len(), 1000);
        assert!(cache.contains_key(&1000));
    }

    #[test]
    fn test_sieve_extend() {
        let mut cache = SIEVE::new(NonZeroUsize::new(5).unwrap());
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
    fn test_sieve_retain_complex() {
        let mut cache = SIEVE::new(NonZeroUsize::new(6).unwrap());
        for i in 1..=6 {
            cache.insert(i, format!("value_{}", i));
        }

        cache.get(&2);
        cache.get(&4);
        cache.get(&6);

        cache.retain(|k, v| *k % 2 == 0 && v.contains("value_"));

        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&4));
        assert!(cache.contains_key(&6));
        assert!(!cache.contains_key(&1));
        assert!(!cache.contains_key(&3));
        assert!(!cache.contains_key(&5));
    }

    #[test]
    fn test_sieve_stress_eviction_patterns() {
        let mut cache = SIEVE::new(NonZeroUsize::new(3).unwrap());

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
    fn test_sieve_clone() {
        let mut cache = SIEVE::new(NonZeroUsize::new(3).unwrap());
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
    fn test_sieve_debug_format() {
        let mut cache = SIEVE::new(NonZeroUsize::new(2).unwrap());
        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("SIEVE"));

        assert!(debug_str.len() > 10);
    }

    #[test]
    fn test_sieve_from_iter_capacity_exceeded() {
        let items: Vec<(i32, String)> = (0..10).map(|i| (i, format!("value_{}", i))).collect();

        let cache: SIEVE<i32, String> = items.into_iter().collect();
        assert_eq!(cache.capacity(), 10);
        assert_eq!(cache.len(), 10);

        for i in 0..10 {
            assert!(cache.contains_key(&i));
        }
    }

    #[test]
    fn test_sieve_iterator_stability_during_modification() {
        let mut cache = SIEVE::new(NonZeroUsize::new(5).unwrap());
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
    fn test_sieve_pop_until_empty() {
        let mut cache = SIEVE::new(NonZeroUsize::new(4).unwrap());
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
}
