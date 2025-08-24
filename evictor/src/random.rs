#[cfg(debug_assertions)]
use rand::seq::SliceRandom;

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
pub struct RandomPolicy;

impl private::Sealed for RandomPolicy {}

#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct RandomEntry<Value> {
    value: Value,
}

impl<Value> private::Sealed for RandomEntry<Value> {}

impl<Value> EntryValue<Value> for RandomEntry<Value> {
    fn new(value: Value) -> Self {
        Self { value }
    }

    fn into_value(self) -> Value {
        self.value
    }

    fn value(&self) -> &Value {
        &self.value
    }

    fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }
}

#[derive(Debug, Clone, Copy, Default)]
#[doc(hidden)]
pub struct RandomMetadata;

impl private::Sealed for RandomMetadata {}

impl<T> Metadata<T> for RandomMetadata {
    type EntryType = RandomEntry<T>;

    fn candidate_removal_index<K>(&self, queue: &LinkedHashMap<K, RandomEntry<T>>) -> Ptr {
        queue
            .index_ptr_unstable(rand::random_range(0..queue.len().max(1)))
            .unwrap_or_default()
    }
}

impl<T> Policy<T> for RandomPolicy {
    type IntoIter<K> = IntoIter<K, T>;
    type MetadataType = RandomMetadata;

    fn insert_or_update_entry<K: std::hash::Hash + Eq>(
        key: K,
        make_room_on_insert: bool,
        get_value: impl FnOnce(&K, /* is_insert */ bool) -> InsertOrUpdateAction<T>,
        metadata: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) -> InsertionResult<K, T> {
        let would_evict = metadata.candidate_removal_index(queue);
        match queue.entry(key) {
            linked_hashmap::Entry::Occupied(mut occupied_entry) => {
                let ptr = occupied_entry.ptr();
                match get_value(occupied_entry.key(), false) {
                    InsertOrUpdateAction::NoInsert(_) => {
                        unreachable!("Cache hit should not return NoInsert");
                    }
                    InsertOrUpdateAction::TouchNoUpdate => {
                        InsertionResult::FoundTouchedNoUpdate(ptr)
                    }
                    InsertOrUpdateAction::InsertOrUpdate(value) => {
                        let entry = occupied_entry.get_mut();
                        *entry.value_mut() = value;
                        InsertionResult::Updated(ptr)
                    }
                }
            }
            linked_hashmap::Entry::Vacant(vacant_entry) => {
                let value = match get_value(vacant_entry.key(), true) {
                    InsertOrUpdateAction::TouchNoUpdate => {
                        unreachable!("Cache miss should not return TouchNoUpdate");
                    }
                    InsertOrUpdateAction::NoInsert(value) => {
                        return InsertionResult::NotFoundNoInsert(vacant_entry.into_key(), value);
                    }
                    InsertOrUpdateAction::InsertOrUpdate(value) => value,
                };
                let ptr = if make_room_on_insert {
                    let ptr = vacant_entry.insert_tail(RandomEntry::new(value));
                    Self::remove_entry(would_evict, metadata, queue);
                    ptr
                } else {
                    vacant_entry.insert_tail(RandomEntry::new(value))
                };
                InsertionResult::Inserted(ptr)
            }
        }
    }

    fn touch_entry<K: std::hash::Hash + Eq>(
        _: Ptr,
        _: &mut Self::MetadataType,
        _: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) {
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
        }) = queue.remove_ptr(ptr)
        else {
            return (Ptr::null(), None);
        };
        (next, Some((key, value)))
    }

    fn remove_key<K: std::hash::Hash + Eq>(
        key: &K,
        _: &mut Self::MetadataType,
        queue: &mut LinkedHashMap<K, <Self::MetadataType as Metadata<T>>::EntryType>,
    ) -> Option<<Self::MetadataType as Metadata<T>>::EntryType> {
        queue.remove(key).map(|removed| removed.1.value)
    }

    fn iter<'q, K>(
        _metadata: &'q Self::MetadataType,
        queue: &'q LinkedHashMap<K, RandomEntry<T>>,
    ) -> impl Iterator<Item = (&'q K, &'q T)>
    where
        T: 'q,
    {
        #[cfg(debug_assertions)]
        let mut order = (0..queue.len()).collect::<Vec<_>>();
        #[cfg(debug_assertions)]
        order.shuffle(&mut rand::rng());

        Iter {
            #[cfg(debug_assertions)]
            index: order.pop().unwrap_or(queue.len()),
            #[cfg(not(debug_assertions))]
            index: 0,
            #[cfg(debug_assertions)]
            order,
            queue,
        }
    }

    fn into_iter<K>(
        _metadata: Self::MetadataType,
        queue: LinkedHashMap<K, RandomEntry<T>>,
    ) -> Self::IntoIter<K> {
        #[cfg(debug_assertions)]
        let mut order = (0..queue.len()).collect::<Vec<_>>();
        #[cfg(debug_assertions)]
        order.shuffle(&mut rand::rng());

        IntoIter {
            #[cfg(debug_assertions)]
            index: order.pop().unwrap_or(queue.len()),
            #[cfg(not(debug_assertions))]
            index: 0,
            #[cfg(debug_assertions)]
            order,
            queue: queue.into_iter().map(Some).collect(),
        }
    }

    fn into_entries<K>(
        _metadata: Self::MetadataType,
        queue: LinkedHashMap<K, RandomEntry<T>>,
    ) -> impl Iterator<Item = (K, RandomEntry<T>)> {
        #[cfg(debug_assertions)]
        let mut order = (0..queue.len()).collect::<Vec<_>>();
        #[cfg(debug_assertions)]
        order.shuffle(&mut rand::rng());

        IntoEntriesIter {
            #[cfg(debug_assertions)]
            index: order.pop().unwrap_or(queue.len()),
            #[cfg(not(debug_assertions))]
            index: 0,
            #[cfg(debug_assertions)]
            order,
            queue: queue.into_iter().map(Some).collect(),
        }
    }

    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    fn debug_validate<K: std::hash::Hash + Eq>(
        _: &Self::MetadataType,
        _: &LinkedHashMap<K, RandomEntry<T>>,
    ) {
        // Nothing to do
    }
}

struct Iter<'q, K, T> {
    #[cfg(debug_assertions)]
    order: Vec<usize>,
    queue: &'q LinkedHashMap<K, RandomEntry<T>>,
    index: usize,
}

impl<'q, K, T> Iterator for Iter<'q, K, T> {
    type Item = (&'q K, &'q T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.queue.len() {
            let ptr = self.queue.index_ptr_unstable(self.index)?;
            #[cfg(debug_assertions)]
            {
                self.index = self.order.pop().unwrap_or(self.queue.len());
            }
            #[cfg(not(debug_assertions))]
            {
                self.index += 1;
            }

            self.queue
                .ptr_get_entry(ptr)
                .map(|(key, entry)| (key, entry.value()))
        } else {
            None
        }
    }
}

#[doc(hidden)]
pub struct IntoIter<K, T> {
    #[cfg(debug_assertions)]
    order: Vec<usize>,
    queue: Vec<Option<(K, RandomEntry<T>)>>,
    index: usize,
}

impl<K, T> Iterator for IntoIter<K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.queue.len() {
            let (key, entry) = self.queue.get_mut(self.index)?.take()?;
            #[cfg(debug_assertions)]
            {
                self.index = self.order.pop().unwrap_or(self.queue.len());
            }
            #[cfg(not(debug_assertions))]
            {
                self.index += 1;
            }
            Some((key, entry.into_value()))
        } else {
            None
        }
    }
}

#[doc(hidden)]
pub struct IntoEntriesIter<K, T> {
    #[cfg(debug_assertions)]
    order: Vec<usize>,
    queue: Vec<Option<(K, RandomEntry<T>)>>,
    index: usize,
}

impl<K, T> Iterator for IntoEntriesIter<K, T> {
    type Item = (K, RandomEntry<T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.queue.len() {
            let (key, entry) = self.queue.get_mut(self.index)?.take()?;
            #[cfg(debug_assertions)]
            {
                self.index = self.order.pop().unwrap_or(self.queue.len());
            }
            #[cfg(not(debug_assertions))]
            {
                self.index += 1;
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

    use ntest::timeout;

    use crate::Random;

    #[test]
    #[timeout(1000)]
    fn test_random_empty_cache() {
        let mut cache = Random::<i32, i32>::new(NonZeroUsize::new(3).unwrap());

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
}
