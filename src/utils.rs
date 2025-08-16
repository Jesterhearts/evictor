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
