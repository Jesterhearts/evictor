#[cfg(debug_assertions)]
use rand::seq::SliceRandom;

use crate::{
    EntryValue,
    Metadata,
    Policy,
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
pub struct RandomMetadata {
    num_entries: usize,
}

impl private::Sealed for RandomMetadata {}

impl Metadata for RandomMetadata {
    fn tail_index(&self) -> usize {
        rand::random::<u64>() as usize % self.num_entries
    }
}

impl<T> Policy<T> for RandomPolicy {
    type EntryType = RandomEntry<T>;
    type IntoIter<K> = IntoIter<K, T>;
    type MetadataType = RandomMetadata;

    fn touch_entry<K>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> usize {
        metadata.num_entries = queue.len();
        index
    }

    fn swap_remove_entry<K: std::hash::Hash + Eq>(
        index: usize,
        metadata: &mut Self::MetadataType,
        queue: &mut indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> Option<(K, Self::EntryType)> {
        let result = queue.swap_remove_index(index);
        metadata.num_entries = queue.len();
        result
    }

    fn iter<'q, K>(
        _metadata: &'q Self::MetadataType,
        queue: &'q indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
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
            order,
            queue,
            index: 0,
        }
    }

    fn into_iter<K>(
        _metadata: Self::MetadataType,
        queue: indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> Self::IntoIter<K> {
        #[cfg(debug_assertions)]
        let mut order = (0..queue.len()).collect::<Vec<_>>();
        #[cfg(debug_assertions)]
        order.shuffle(&mut rand::rng());

        IntoIter {
            #[cfg(debug_assertions)]
            order,
            queue: queue.into_iter().map(Some).collect(),
            index: 0,
        }
    }

    fn into_entries<K>(
        _metadata: Self::MetadataType,
        queue: indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
    ) -> impl Iterator<Item = (K, Self::EntryType)> {
        #[cfg(debug_assertions)]
        let mut order = (0..queue.len()).collect::<Vec<_>>();
        #[cfg(debug_assertions)]
        order.shuffle(&mut rand::rng());

        IntoEntriesIter {
            #[cfg(debug_assertions)]
            order,
            queue: queue.into_iter().map(Some).collect(),
            index: 0,
        }
    }
}

struct Iter<'q, K, T> {
    #[cfg(debug_assertions)]
    order: Vec<usize>,
    queue: &'q indexmap::IndexMap<K, RandomEntry<T>, crate::RandomState>,
    index: usize,
}

impl<'q, K, T> Iterator for Iter<'q, K, T> {
    type Item = (&'q K, &'q T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.queue.len() {
            let (key, entry) = self.queue.get_index(self.index)?;
            #[cfg(debug_assertions)]
            {
                self.index = self.order.pop().unwrap_or(self.queue.len());
            }
            #[cfg(not(debug_assertions))]
            {
                self.index += 1;
            }
            Some((key, entry.value()))
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
