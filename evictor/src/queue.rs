macro_rules! impl_queue_policy {
    (($policy_name:ident, $entry_name:ident, $metadata_name:ident) => $link:ident ) => {
        use crate::{
            EntryValue,
            Metadata,
            Policy,
            private,
            utils::{
                impl_ll_iters,
                swap_remove_ll_entry,
            },
        };

        #[derive(Debug, Clone, Copy)]
        #[doc(hidden)]
        pub struct $policy_name;

        impl private::Sealed for $policy_name {}

        #[derive(Debug, Clone, Copy)]
        #[doc(hidden)]
        pub struct $entry_name<Value> {
            value: Value,
            next: Option<usize>,
            prev: Option<usize>,
        }

        impl<Value> private::Sealed for $entry_name<Value> {}

        impl<Value> EntryValue<Value> for $entry_name<Value> {
            fn new(value: Value) -> Self {
                $entry_name {
                    value,
                    next: None,
                    prev: None,
                }
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
        pub struct $metadata_name {
            head: usize,
            tail: usize,
        }

        impl private::Sealed for $metadata_name {}

        impl Metadata for $metadata_name {
            fn tail_index(&self) -> usize {
                self.tail
            }
        }

        impl<T> Policy<T> for $policy_name {
            type EntryType = $entry_name<T>;
            type IntoIter<K> = IntoIter<K, T>;
            type MetadataType = $metadata_name;

            fn touch_entry<K>(
                index: usize,
                metadata: &mut Self::MetadataType,
                queue: &mut indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
            ) -> usize {
                if index >= queue.len() {
                    return index;
                }

                // In queues, touching an entry does not change its position, but we still need
                // to link it into the list if it has no links.
                if queue[index].next.is_none() && queue[index].prev.is_none() {
                    $link!(metadata, index, queue);
                }

                index
            }

            fn swap_remove_entry<K: std::hash::Hash + Eq>(
                index: usize,
                metadata: &mut Self::MetadataType,
                queue: &mut indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
            ) -> Option<(K, Self::EntryType)> {
                swap_remove_ll_entry!(index, metadata, queue)
            }

            fn iter<'q, K>(
                metadata: &'q Self::MetadataType,
                queue: &'q indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
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
                queue: indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
            ) -> Self::IntoIter<K> {
                IntoIter {
                    queue: queue.into_iter().map(Some).collect(),
                    index: Some(metadata.tail),
                }
            }

            fn into_entries<K>(
                metadata: Self::MetadataType,
                queue: indexmap::IndexMap<K, Self::EntryType, crate::RandomState>,
            ) -> impl Iterator<Item = (K, Self::EntryType)> {
                IntoEntriesIter {
                    queue: queue.into_iter().map(Some).collect(),
                    index: Some(metadata.tail),
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

        impl_ll_iters!($entry_name);
    };
}

pub(crate) mod fifo {
    macro_rules! link_as_head {
        ($metadata:ident, $index:ident, $queue:ident) => {
            if $metadata.head == $index {
                return $index;
            }

            $queue[$index].prev = Some($metadata.head);
            $queue[$metadata.head].next = Some($index);
            $metadata.head = $index;
        };
    }

    impl_queue_policy!(
        (FIFOPolicy, FIFOEntry, FIFOMetadata) =>  link_as_head
    );
}

pub use fifo::FIFOPolicy;

pub(crate) mod lifo {
    macro_rules! link_as_tail {
        ($metadata:ident, $index:ident, $queue:ident) => {
            if $metadata.tail == $index {
                return $index;
            }

            $queue[$index].next = Some($metadata.tail);
            $queue[$metadata.tail].prev = Some($index);
            $metadata.tail = $index;
        };
    }

    impl_queue_policy!(
        (LIFOPolicy, LIFOEntry, LIFOMetadata) => link_as_tail
    );
}

pub use lifo::LIFOPolicy;

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::{
        FIFO,
        LIFO,
    };

    #[test]
    fn test_fifo_trivial() {
        let mut fifo = FIFO::new(NonZeroUsize::new(3).unwrap());
        fifo.insert("a", 1);
        fifo.insert("b", 2);
        fifo.insert("c", 3);

        assert_eq!(fifo.get(&"a"), Some(&1));
        assert_eq!(fifo.get(&"b"), Some(&2));
        assert_eq!(fifo.get(&"c"), Some(&3));

        fifo.get(&"a");
        fifo.insert("d", 4);

        assert_eq!(fifo.get(&"a"), None);
        assert_eq!(fifo.get(&"b"), Some(&2));
    }

    #[test]
    fn test_fifo_eviction_order() {
        let mut fifo = FIFO::new(NonZeroUsize::new(3).unwrap());

        fifo.insert(1, 10);
        fifo.insert(2, 20);
        fifo.insert(3, 30);

        fifo.insert(4, 40);

        assert_eq!(
            fifo.into_iter().collect::<Vec<_>>(),
            [(2, 20), (3, 30), (4, 40)]
        );
    }

    #[test]
    fn test_fifo_access_does_not_update_order() {
        let mut fifo = FIFO::new(NonZeroUsize::new(3).unwrap());

        fifo.insert(1, 10);
        fifo.insert(2, 20);
        fifo.insert(3, 30);

        fifo.get(&1);

        fifo.insert(4, 40);

        assert_eq!(
            fifo.into_iter().collect::<Vec<_>>(),
            [(2, 20), (3, 30), (4, 40)]
        );
    }

    #[test]
    fn test_fifo_update_existing_key() {
        let mut fifo = FIFO::new(NonZeroUsize::new(2).unwrap());

        fifo.insert(1, 10);
        fifo.insert(2, 20);

        fifo.insert(2, 200);

        fifo.insert(3, 30);

        assert_eq!(fifo.get(&1), None);
        assert_eq!(fifo.get(&2), Some(&200));
        assert_eq!(fifo.get(&3), Some(&30));
    }

    #[test]
    fn test_fifo_single_capacity() {
        let mut fifo = FIFO::new(NonZeroUsize::new(1).unwrap());

        fifo.insert(1, 10);
        assert_eq!(fifo.get(&1), Some(&10));

        fifo.insert(2, 20);
        assert_eq!(fifo.get(&1), None);
        assert_eq!(fifo.get(&2), Some(&20));

        fifo.get(&2);
        fifo.insert(3, 30);
        assert_eq!(fifo.get(&2), None);
        assert_eq!(fifo.get(&3), Some(&30));
    }

    #[test]
    fn test_fifo_get_or_insert_with_eviction() {
        let mut fifo = FIFO::new(NonZeroUsize::new(2).unwrap());

        fifo.insert(1, 10);
        fifo.insert(2, 20);

        fifo.get(&1);

        let value = fifo.get_or_insert_with(3, |_| 30);
        assert_eq!(value, &30);

        assert_eq!(fifo.into_iter().collect::<Vec<_>>(), [(2, 20), (3, 30)]);
    }

    #[test]
    fn test_fifo_get_or_insert_existing_key() {
        let mut fifo = FIFO::new(NonZeroUsize::new(3).unwrap());

        fifo.insert(1, 10);
        fifo.insert(2, 20);
        fifo.insert(3, 30);

        let value = fifo.get_or_insert_with(1, |_| 999);
        assert_eq!(value, &10);

        fifo.insert(4, 40);

        assert_eq!(
            fifo.into_iter().collect::<Vec<_>>(),
            [(2, 20), (3, 30), (4, 40),]
        );
    }

    #[test]
    fn test_fifo_empty_cache() {
        let mut fifo = FIFO::<i32, i32>::new(NonZeroUsize::new(3).unwrap());

        assert!(fifo.is_empty());
        assert_eq!(fifo.len(), 0);
        assert_eq!(fifo.capacity(), 3);
        assert_eq!(fifo.get(&1), None);
        assert_eq!(fifo.peek(&1), None);
        assert_eq!(fifo.remove(&1), None);
        assert_eq!(fifo.pop(), None);
        assert!(fifo.tail().is_none());
        assert!(!fifo.contains_key(&1));
    }

    #[test]
    fn test_fifo_capacity_constraints() {
        let fifo = FIFO::<i32, i32>::new(NonZeroUsize::new(5).unwrap());
        assert_eq!(fifo.capacity(), 5);

        let fifo = FIFO::<i32, i32>::new(NonZeroUsize::new(1).unwrap());
        assert_eq!(fifo.capacity(), 1);

        let fifo = FIFO::<i32, i32>::new(NonZeroUsize::new(100).unwrap());
        assert_eq!(fifo.capacity(), 100);
    }

    #[test]
    fn test_fifo_remove() {
        let mut fifo = FIFO::new(NonZeroUsize::new(3).unwrap());
        fifo.insert(1, 10);
        fifo.insert(2, 20);
        fifo.insert(3, 30);

        assert_eq!(fifo.remove(&2), Some(20));
        assert_eq!(fifo.len(), 2);
        assert!(!fifo.contains_key(&2));
        assert!(fifo.contains_key(&1));
        assert!(fifo.contains_key(&3));

        assert_eq!(fifo.remove(&2), None);
        assert_eq!(fifo.len(), 2);

        fifo.insert(4, 40);
        assert_eq!(fifo.len(), 3);

        assert_eq!(
            fifo.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            [(1, 10), (3, 30), (4, 40)]
        );

        fifo.insert(5, 50);

        assert_eq!(
            fifo.into_iter().collect::<Vec<_>>(),
            [(3, 30), (4, 40), (5, 50)]
        );
    }

    #[test]
    fn test_fifo_pop() {
        let mut fifo = FIFO::new(NonZeroUsize::new(3).unwrap());
        fifo.insert(1, 10);
        fifo.insert(2, 20);
        fifo.insert(3, 30);

        let popped = fifo.pop();
        assert_eq!(popped, Some((1, 10)));
        assert_eq!(fifo.len(), 2);
        assert!(!fifo.contains_key(&1));

        let popped = fifo.pop();
        assert_eq!(popped, Some((2, 20)));
        assert_eq!(fifo.len(), 1);

        let popped = fifo.pop();
        assert_eq!(popped, Some((3, 30)));
        assert_eq!(fifo.len(), 0);
        assert!(fifo.is_empty());

        let popped = fifo.pop();
        assert_eq!(popped, None);

        assert_eq!(fifo.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    fn test_fifo_tail() {
        let mut fifo = FIFO::new(NonZeroUsize::new(3).unwrap());

        assert!(fifo.tail().is_none());

        fifo.insert(1, 10);
        assert_eq!(fifo.tail(), Some((&1, &10)));

        fifo.insert(2, 20);
        assert_eq!(fifo.tail(), Some((&1, &10)));

        fifo.insert(3, 30);
        assert_eq!(fifo.tail(), Some((&1, &10)));

        fifo.get(&1);
        assert_eq!(fifo.tail(), Some((&1, &10)));
    }

    #[test]
    fn test_fifo_clear() {
        let mut fifo = FIFO::new(NonZeroUsize::new(3).unwrap());
        fifo.insert(1, 10);
        fifo.insert(2, 20);
        fifo.insert(3, 30);

        assert_eq!(fifo.len(), 3);
        assert!(!fifo.is_empty());

        fifo.clear();

        assert_eq!(fifo.len(), 0);
        assert!(fifo.is_empty());
        assert_eq!(fifo.capacity(), 3);
        assert!(fifo.tail().is_none());

        assert_eq!(fifo.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    fn test_fifo_mutable_access() {
        let mut fifo = FIFO::new(NonZeroUsize::new(2).unwrap());
        fifo.insert(1, String::from("hello"));
        fifo.insert(2, String::from("world"));

        if let Some(val) = fifo.get_mut(&1) {
            val.push_str(" modified");
        }

        assert_eq!(fifo.get(&1), Some(&String::from("hello modified")));

        fifo.insert(3, String::from("new"));

        assert_eq!(
            fifo.into_iter().collect::<Vec<_>>(),
            [(2, String::from("world")), (3, String::from("new"))]
        );
    }

    #[test]
    fn test_fifo_insert_mut() {
        let mut fifo = FIFO::new(NonZeroUsize::new(2).unwrap());

        let val = fifo.insert_mut(1, String::from("test"));
        val.push_str(" modified");

        assert_eq!(fifo.get(&1), Some(&String::from("test modified")));

        let val = fifo.insert_mut(1, String::from("replaced"));
        val.push_str(" again");

        assert_eq!(fifo.get(&1), Some(&String::from("replaced again")));
    }

    #[test]
    fn test_fifo_get_or_insert_with_mut() {
        let mut fifo = FIFO::new(NonZeroUsize::new(2).unwrap());
        fifo.insert(1, String::from("existing"));

        let val = fifo.get_or_insert_with_mut(1, |_| String::from("new"));
        val.push_str(" modified");

        assert_eq!(fifo.get(&1), Some(&String::from("existing modified")));

        let val = fifo.get_or_insert_with_mut(2, |_| String::from("created"));
        val.push_str(" also modified");

        assert_eq!(fifo.get(&2), Some(&String::from("created also modified")));
    }

    #[test]
    fn test_fifo_extend() {
        let mut fifo = FIFO::new(NonZeroUsize::new(5).unwrap());
        fifo.insert(1, 10);

        let items = vec![(2, 20), (3, 30), (4, 40)];
        fifo.extend(items);

        assert_eq!(fifo.len(), 4);

        assert_eq!(
            fifo.into_iter().collect::<Vec<_>>(),
            [(1, 10), (2, 20), (3, 30), (4, 40)]
        );
    }

    #[test]
    fn test_fifo_from_iterator() {
        let items = vec![(1, 10), (2, 20), (3, 30)];
        let fifo: FIFO<i32, i32> = items.into_iter().collect();

        assert_eq!(fifo.len(), 3);
        assert_eq!(fifo.capacity(), 3);

        assert_eq!(
            fifo.into_iter().collect::<Vec<_>>(),
            [(1, 10), (2, 20), (3, 30)]
        );
    }

    #[test]
    fn test_fifo_shrink_to_fit() {
        let mut fifo = FIFO::new(NonZeroUsize::new(10).unwrap());
        fifo.insert(1, 10);
        fifo.insert(2, 20);

        fifo.shrink_to_fit();

        assert_eq!(fifo.len(), 2);

        assert_eq!(fifo.into_iter().collect::<Vec<_>>(), [(1, 10), (2, 20)]);
    }

    #[test]
    fn test_fifo_retain_basic() {
        let mut fifo = FIFO::new(NonZeroUsize::new(5).unwrap());

        fifo.insert(1, "one");
        fifo.insert(2, "two");
        fifo.insert(3, "three");
        fifo.insert(4, "four");
        fifo.insert(5, "five");

        fifo.retain(|&k, _| k % 2 == 0);

        assert_eq!(fifo.len(), 2);

        assert_eq!(
            fifo.into_iter().collect::<Vec<_>>(),
            [(2, "two"), (4, "four")]
        );
    }

    #[test]
    fn test_fifo_peek_mut_no_modification() {
        let mut cache = FIFO::new(NonZeroUsize::new(3).unwrap());
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
    fn test_lifo_trivial() {
        let mut lifo = LIFO::new(NonZeroUsize::new(3).unwrap());
        lifo.insert("a", 1);
        lifo.insert("b", 2);
        lifo.insert("c", 3);

        assert_eq!(lifo.get(&"a"), Some(&1));
        assert_eq!(lifo.get(&"b"), Some(&2));
        assert_eq!(lifo.get(&"c"), Some(&3));

        lifo.get(&"a");
        lifo.insert("d", 4);

        assert_eq!(lifo.get(&"c"), None);
        assert_eq!(lifo.get(&"b"), Some(&2));
    }

    #[test]
    fn test_lifo_eviction_order() {
        let mut lifo = LIFO::new(NonZeroUsize::new(3).unwrap());

        lifo.insert(1, 10);
        lifo.insert(2, 20);
        lifo.insert(3, 30);

        lifo.insert(4, 40);

        assert_eq!(
            lifo.into_iter().collect::<Vec<_>>(),
            [(4, 40), (2, 20), (1, 10),]
        );
    }

    #[test]
    fn test_lifo_access_does_not_update_order() {
        let mut lifo = LIFO::new(NonZeroUsize::new(3).unwrap());

        lifo.insert(1, 10);
        lifo.insert(2, 20);
        lifo.insert(3, 30);

        lifo.get(&1);

        lifo.insert(4, 40);

        assert_eq!(
            lifo.into_iter().collect::<Vec<_>>(),
            [(4, 40), (2, 20), (1, 10),]
        );
    }

    #[test]
    fn test_lifo_update_existing_key() {
        let mut lifo = LIFO::new(NonZeroUsize::new(2).unwrap());

        lifo.insert(1, 10);
        lifo.insert(2, 20);

        lifo.insert(1, 100);

        lifo.insert(3, 30);

        assert_eq!(lifo.get(&1), Some(&100));
        assert_eq!(lifo.get(&3), Some(&30));
        assert_eq!(lifo.get(&2), None);
    }

    #[test]
    fn test_lifo_single_capacity() {
        let mut lifo = LIFO::new(NonZeroUsize::new(1).unwrap());

        lifo.insert(1, 10);
        assert_eq!(lifo.get(&1), Some(&10));

        lifo.insert(2, 20);
        assert_eq!(lifo.get(&1), None);
        assert_eq!(lifo.get(&2), Some(&20));

        lifo.get(&2);
        lifo.insert(3, 30);
        assert_eq!(lifo.get(&2), None);
        assert_eq!(lifo.get(&3), Some(&30));
    }

    #[test]
    fn test_lifo_get_or_insert_with_eviction() {
        let mut lifo = LIFO::new(NonZeroUsize::new(2).unwrap());

        lifo.insert(1, 10);
        lifo.insert(2, 20);

        lifo.get(&1);

        let value = lifo.get_or_insert_with(3, |_| 30);
        assert_eq!(value, &30);

        assert_eq!(lifo.into_iter().collect::<Vec<_>>(), [(3, 30), (1, 10),]);
    }

    #[test]
    fn test_lifo_get_or_insert_existing_key() {
        let mut lifo = LIFO::new(NonZeroUsize::new(3).unwrap());

        lifo.insert(1, 10);
        lifo.insert(2, 20);
        lifo.insert(3, 30);

        let value = lifo.get_or_insert_with(1, |_| 999);
        assert_eq!(value, &10);

        lifo.insert(4, 40);

        assert_eq!(
            lifo.into_iter().collect::<Vec<_>>(),
            [(4, 40), (2, 20), (1, 10),]
        );
    }

    #[test]
    fn test_lifo_empty_cache() {
        let mut lifo = LIFO::<i32, i32>::new(NonZeroUsize::new(3).unwrap());

        assert!(lifo.is_empty());
        assert_eq!(lifo.len(), 0);
        assert_eq!(lifo.capacity(), 3);
        assert_eq!(lifo.get(&1), None);
        assert_eq!(lifo.peek(&1), None);
        assert_eq!(lifo.remove(&1), None);
        assert_eq!(lifo.pop(), None);
        assert!(lifo.tail().is_none());
        assert!(!lifo.contains_key(&1));
    }

    #[test]
    fn test_lifo_capacity_constraints() {
        let lifo = LIFO::<i32, i32>::new(NonZeroUsize::new(5).unwrap());
        assert_eq!(lifo.capacity(), 5);

        let lifo = LIFO::<i32, i32>::new(NonZeroUsize::new(1).unwrap());
        assert_eq!(lifo.capacity(), 1);

        let lifo = LIFO::<i32, i32>::new(NonZeroUsize::new(100).unwrap());
        assert_eq!(lifo.capacity(), 100);
    }

    #[test]
    fn test_lifo_remove() {
        let mut lifo = LIFO::new(NonZeroUsize::new(3).unwrap());
        lifo.insert(1, 10);
        lifo.insert(2, 20);
        lifo.insert(3, 30);

        assert_eq!(lifo.remove(&2), Some(20));
        assert_eq!(lifo.len(), 2);
        assert!(!lifo.contains_key(&2));
        assert!(lifo.contains_key(&1));
        assert!(lifo.contains_key(&3));

        assert_eq!(lifo.remove(&2), None);
        assert_eq!(lifo.len(), 2);

        lifo.insert(4, 40);
        assert_eq!(lifo.len(), 3);

        assert_eq!(
            lifo.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            [(4, 40), (3, 30), (1, 10),]
        );

        lifo.insert(5, 50);

        assert_eq!(
            lifo.into_iter().collect::<Vec<_>>(),
            [(5, 50), (3, 30), (1, 10)]
        );
    }

    #[test]
    fn test_lifo_pop() {
        let mut lifo = LIFO::new(NonZeroUsize::new(3).unwrap());
        lifo.insert(1, 10);
        lifo.insert(2, 20);
        lifo.insert(3, 30);

        let popped = lifo.pop();
        assert_eq!(popped, Some((3, 30)));
        assert_eq!(lifo.len(), 2);
        assert!(!lifo.contains_key(&3));

        let popped = lifo.pop();
        assert_eq!(popped, Some((2, 20)));
        assert_eq!(lifo.len(), 1);

        let popped = lifo.pop();
        assert_eq!(popped, Some((1, 10)));
        assert_eq!(lifo.len(), 0);
        assert!(lifo.is_empty());

        let popped = lifo.pop();
        assert_eq!(popped, None);

        assert_eq!(lifo.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    fn test_lifo_tail() {
        let mut lifo = LIFO::new(NonZeroUsize::new(3).unwrap());

        assert!(lifo.tail().is_none());

        lifo.insert(1, 10);
        assert_eq!(lifo.tail(), Some((&1, &10)));

        lifo.insert(2, 20);
        assert_eq!(lifo.tail(), Some((&2, &20)));

        lifo.insert(3, 30);
        assert_eq!(lifo.tail(), Some((&3, &30)));

        lifo.get(&1);
        assert_eq!(lifo.tail(), Some((&3, &30)));
    }

    #[test]
    fn test_lifo_clear() {
        let mut lifo = LIFO::new(NonZeroUsize::new(3).unwrap());
        lifo.insert(1, 10);
        lifo.insert(2, 20);
        lifo.insert(3, 30);

        assert_eq!(lifo.len(), 3);
        assert!(!lifo.is_empty());

        lifo.clear();

        assert_eq!(lifo.len(), 0);
        assert!(lifo.is_empty());
        assert_eq!(lifo.capacity(), 3);
        assert!(lifo.tail().is_none());

        assert_eq!(lifo.into_iter().collect::<Vec<_>>(), []);
    }

    #[test]
    fn test_lifo_mutable_access() {
        let mut lifo = LIFO::new(NonZeroUsize::new(2).unwrap());
        lifo.insert(1, String::from("hello"));
        lifo.insert(2, String::from("world"));

        if let Some(val) = lifo.get_mut(&1) {
            val.push_str(" modified");
        }

        assert_eq!(lifo.get(&1), Some(&String::from("hello modified")));

        lifo.insert(3, String::from("new"));

        assert_eq!(
            lifo.into_iter().collect::<Vec<_>>(),
            [
                (3, String::from("new")),
                (1, String::from("hello modified")),
            ]
        );
    }

    #[test]
    fn test_lifo_insert_mut() {
        let mut lifo = LIFO::new(NonZeroUsize::new(2).unwrap());

        let val = lifo.insert_mut(1, String::from("test"));
        val.push_str(" modified");

        assert_eq!(lifo.get(&1), Some(&String::from("test modified")));

        let val = lifo.insert_mut(1, String::from("replaced"));
        val.push_str(" again");

        assert_eq!(lifo.get(&1), Some(&String::from("replaced again")));
    }

    #[test]
    fn test_lifo_get_or_insert_with_mut() {
        let mut lifo = LIFO::new(NonZeroUsize::new(2).unwrap());
        lifo.insert(1, String::from("existing"));

        let val = lifo.get_or_insert_with_mut(1, |_| String::from("new"));
        val.push_str(" modified");

        assert_eq!(lifo.get(&1), Some(&String::from("existing modified")));

        let val = lifo.get_or_insert_with_mut(2, |_| String::from("created"));
        val.push_str(" also modified");

        assert_eq!(lifo.get(&2), Some(&String::from("created also modified")));
    }

    #[test]
    fn test_lifo_extend() {
        let mut lifo = LIFO::new(NonZeroUsize::new(5).unwrap());
        lifo.insert(1, 10);

        let items = vec![(2, 20), (3, 30), (4, 40)];
        lifo.extend(items);

        assert_eq!(lifo.len(), 4);

        assert_eq!(
            lifo.into_iter().collect::<Vec<_>>(),
            [(4, 40), (3, 30), (2, 20), (1, 10),]
        );
    }

    #[test]
    fn test_lifo_from_iterator() {
        let items = vec![(1, 10), (2, 20), (3, 30)];
        let lifo: LIFO<i32, i32> = items.into_iter().collect();

        assert_eq!(lifo.len(), 3);
        assert_eq!(lifo.capacity(), 3);

        assert_eq!(
            lifo.into_iter().collect::<Vec<_>>(),
            [(3, 30), (2, 20), (1, 10),]
        );
    }

    #[test]
    fn test_lifo_shrink_to_fit() {
        let mut lifo = LIFO::new(NonZeroUsize::new(10).unwrap());
        lifo.insert(1, 10);
        lifo.insert(2, 20);

        lifo.shrink_to_fit();

        assert_eq!(lifo.len(), 2);

        assert_eq!(lifo.into_iter().collect::<Vec<_>>(), [(2, 20), (1, 10),]);
    }

    #[test]
    fn test_lifo_retain_basic() {
        let mut lifo = LIFO::new(NonZeroUsize::new(5).unwrap());

        lifo.insert(1, "one");
        lifo.insert(2, "two");
        lifo.insert(3, "three");
        lifo.insert(4, "four");
        lifo.insert(5, "five");

        lifo.retain(|&k, _| k % 2 == 0);

        assert_eq!(lifo.len(), 2);

        assert_eq!(
            lifo.into_iter().collect::<Vec<_>>(),
            [(4, "four"), (2, "two"),]
        );
    }

    #[test]
    fn test_lifo_peek_mut_no_modification() {
        let mut cache = LIFO::new(NonZeroUsize::new(3).unwrap());
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
}
