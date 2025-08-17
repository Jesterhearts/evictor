macro_rules! swap_remove_ll_entry {
    ($index:expr, $metadata:expr, $queue:expr) => {{
        if $index >= $queue.len() {
            return None;
        }

        if $queue.len() == 1 {
            return $queue.swap_remove_index($index);
        }

        let (k, e) = $queue.swap_remove_index($index).unwrap();
        if $queue.len() == 1 {
            $metadata.head = 0;
            $metadata.tail = 0;
            $queue[0].prev = None;
            $queue[0].next = None;
            return Some((k, e));
        }

        if $index == $metadata.head {
            // Head was removed, update it
            $metadata.head = e.prev.unwrap_or_default();
        }
        if $metadata.head == $queue.len() {
            // Head was pointing to the perturbed index, or we removed an item from the
            // cache, and it was pointing to the last item in the queue, which is
            // now at $index
            // The second case can happen trivially like this:
            // 1. We have a queue of 2 items, 0 (head) <-> 1 (tail)
            // 2. We remove item 0 manually, which has a prev of item 1, now we have head =
            //    1, tail = 1, and the queue is [2, nul (head, tail)], head needs to point
            //    to 0 (the removed index).
            $metadata.head = $index;
        }

        if $index == $metadata.tail {
            $metadata.tail = e.next.unwrap_or_default();
        }
        if $metadata.tail == $queue.len() {
            $metadata.tail = $index;
        }

        if let Some(prev) = e.prev {
            let len = $queue.len();
            // If the previous pointer of the removed entry points to the last
            // item in the queue, we are really updating the item at $index
            // which replaced it.
            let next = if prev == len {
                &mut $queue[$index].next
            } else {
                &mut $queue[prev].next
            };
            // If the next pointer of the removed entry points to the last
            // item in the queue, we need it to point at $index, which is where that last
            // item moved to.
            if e.next == Some(len) {
                *next = Some($index);
            } else {
                *next = e.next;
            }
        }
        if let Some(next) = e.next {
            let len = $queue.len();
            let prev = if next == len {
                &mut $queue[$index].prev
            } else {
                &mut $queue[next].prev
            };
            if e.prev == Some(len) {
                *prev = Some($index);
            } else {
                *prev = e.prev;
            }
        }

        if $index == $queue.len() {
            return Some((k, e));
        }

        if let Some(next) = $queue[$index].next {
            $queue[next].prev = Some($index);
        }

        if let Some(prev) = $queue[$index].prev {
            $queue[prev].next = Some($index);
        }

        Some((k, e))
    }};
}

pub(crate) use swap_remove_ll_entry;

macro_rules! impl_ll_iters {
    ($entry_t:ident) => {
        struct Iter<'q, K, T> {
            queue: &'q indexmap::IndexMap<K, $entry_t<T>, crate::RandomState>,
            index: Option<usize>,
        }

        impl<'q, K, T> Iterator for Iter<'q, K, T> {
            type Item = (&'q K, &'q T);

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(index) = self.index {
                    let (key, entry) = self.queue.get_index(index)?;
                    self.index = entry.next;
                    Some((key, entry.value()))
                } else {
                    None
                }
            }
        }

        #[doc(hidden)]
        pub struct IntoIter<K, T> {
            queue: Vec<Option<(K, $entry_t<T>)>>,
            index: Option<usize>,
        }

        impl<K, T> Iterator for IntoIter<K, T> {
            type Item = (K, T);

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(index) = self.index {
                    let (key, entry) = self.queue.get_mut(index)?.take()?;
                    self.index = entry.next;
                    Some((key, entry.into_value()))
                } else {
                    None
                }
            }
        }

        struct IntoEntriesIter<K, T> {
            queue: Vec<Option<(K, $entry_t<T>)>>,
            index: Option<usize>,
        }

        impl<K, T> Iterator for IntoEntriesIter<K, T> {
            type Item = (K, $entry_t<T>);

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(index) = self.index {
                    let (key, entry) = self.queue.get_mut(index)?.take()?;
                    self.index = entry.next;
                    Some((key, entry))
                } else {
                    None
                }
            }
        }
    };
}

pub(crate) use impl_ll_iters;
