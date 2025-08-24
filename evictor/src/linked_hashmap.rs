use std::{
    hash::Hash,
    ops::{
        Index,
        IndexMut,
    },
};

use indexmap::{
    IndexMap,
    map,
};

use crate::RandomState;

#[derive(Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
#[repr(transparent)]
pub struct Ptr(usize);

impl std::fmt::Debug for Ptr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Ptr::null() {
            write!(f, "Ptr(null)")
        } else {
            write!(f, "Ptr({})", self.0)
        }
    }
}

impl Default for Ptr {
    fn default() -> Self {
        Ptr::null()
    }
}

impl Ptr {
    pub fn null() -> Self {
        Ptr(usize::MAX)
    }

    pub fn is_null(&self) -> bool {
        *self == Ptr::null()
    }

    pub fn unchecked_from(index: usize) -> Self {
        debug_assert_ne!(index, usize::MAX, "Index must not be usize::MAX");
        Ptr(index)
    }

    pub fn unchecked_get(self) -> usize {
        self.0
    }

    pub fn get(self) -> Option<usize> {
        if self.is_null() {
            None
        } else {
            Some(self.unchecked_get())
        }
    }

    pub fn optional(self) -> Option<Ptr> {
        if self.is_null() { None } else { Some(self) }
    }

    pub fn or(&self, tail_ptr: Ptr) -> Ptr {
        if self.is_null() { tail_ptr } else { *self }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Link {
    prev: Ptr,
    next: Ptr,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct LinkedHashMap<K, T> {
    head: Ptr,
    tail: Ptr,
    links: Vec<Link>,
    map: IndexMap<K, T, RandomState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct RemovedEntry<K, T> {
    pub key: K,
    pub value: T,
    pub prev: Ptr,
    pub next: Ptr,
    pub invalidated: Ptr,
}

impl<K: std::fmt::Debug, T: std::fmt::Debug> std::fmt::Debug for LinkedHashMap<K, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Entry<'a, K, V> {
            key: &'a K,
            value: &'a V,
            previous: Option<&'a K>,
            next: Option<&'a K>,
        }

        let mut entries = Vec::with_capacity(self.len());

        for (index, (k, value)) in self.map.iter().enumerate() {
            let link = &self.links[index];
            let next_key = self.key_for_ptr(link.next);
            let prev_key = self.key_for_ptr(link.prev);

            entries.push(Entry {
                key: k,
                value,
                previous: prev_key,
                next: next_key,
            });
        }

        f.debug_struct("LinkedHashMap")
            .field("len", &self.len())
            .field("head", &self.key_for_ptr(self.head))
            .field("tail", &self.key_for_ptr(self.tail))
            .field("entries", &entries)
            .finish()?;

        Ok(())
    }
}

impl<K, T> Default for LinkedHashMap<K, T> {
    fn default() -> Self {
        LinkedHashMap {
            head: Ptr::null(),
            tail: Ptr::null(),
            links: Vec::new(),
            map: IndexMap::default(),
        }
    }
}

impl<K, T> LinkedHashMap<K, T> {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity < usize::MAX - 1, "Capacity too large");
        LinkedHashMap {
            head: Ptr::null(),
            tail: Ptr::null(),
            links: Vec::with_capacity(capacity),
            map: IndexMap::with_capacity_and_hasher(capacity, RandomState::default()),
        }
    }

    pub fn index_ptr_unstable(&self, index: usize) -> Option<Ptr> {
        Some(Ptr::unchecked_from(index))
    }

    pub fn ptr_index_unstable(&self, ptr: Ptr) -> Option<usize> {
        if ptr.is_null() {
            return None;
        }
        Some(ptr.unchecked_get())
    }

    pub fn move_after(&mut self, moved: Ptr, after: Ptr) -> Option<()> {
        if moved == after {
            return None;
        }

        let after_index = after.get()?;
        let moved_index = moved.get()?;

        let [moved_links, after_links] = &mut self
            .links
            .get_disjoint_mut([moved_index, after_index])
            .ok()?;

        if after_links.next == moved {
            return None;
        }
        let needs_next = moved_links.prev;
        let needs_prev = moved_links.next;
        let also_needs_prev = after_links.next;

        moved_links.next = after_links.next;
        moved_links.prev = after;
        after_links.next = moved;

        if let Some(needs_prev) = self.links.get_mut(also_needs_prev.unchecked_get()) {
            needs_prev.prev = moved;
        }

        if let Some(needs_next) = self.links.get_mut(needs_next.unchecked_get()) {
            needs_next.next = needs_prev;
        }
        if let Some(needs_prev) = self.links.get_mut(needs_prev.unchecked_get()) {
            needs_prev.prev = needs_next;
        }

        if self.head == moved {
            self.head = needs_prev;
        }
        if self.tail == moved {
            self.tail = needs_next;
        }

        if self.tail == after {
            self.tail = moved;
        }

        Some(())
    }

    pub fn move_before(&mut self, moved: Ptr, before: Ptr) -> Option<()> {
        if moved == before {
            return None;
        }

        let before_index = before.get()?;
        let moved_index = moved.get()?;

        let [moved_links, before_links] = &mut self
            .links
            .get_disjoint_mut([moved_index, before_index])
            .ok()?;

        if before_links.prev == moved {
            return None;
        }
        let needs_next = moved_links.prev;
        let needs_prev = moved_links.next;
        let also_needs_next = before_links.prev;

        moved_links.prev = before_links.prev;
        moved_links.next = before;
        before_links.prev = moved;

        if let Some(needs_next) = self.links.get_mut(also_needs_next.unchecked_get()) {
            needs_next.next = moved;
        }

        if let Some(needs_next) = self.links.get_mut(needs_next.unchecked_get()) {
            needs_next.next = needs_prev;
        }
        if let Some(needs_prev) = self.links.get_mut(needs_prev.unchecked_get()) {
            needs_prev.prev = needs_next;
        }

        if self.head == moved {
            self.head = needs_prev;
        }
        if self.tail == moved {
            self.tail = needs_next;
        }

        if self.head == before {
            self.head = moved;
        }

        Some(())
    }

    pub fn link_as_head(&mut self, ptr: Ptr) -> Option<()> {
        let old_head = self.head;
        self.link_node(ptr, Ptr::null(), old_head)
    }

    pub fn link_as_tail(&mut self, ptr: Ptr) -> Option<()> {
        let old_tail = self.tail;
        self.link_node(ptr, old_tail, Ptr::null())
    }

    pub fn link_node(&mut self, ptr: Ptr, prev: Ptr, next: Ptr) -> Option<()> {
        debug_assert_ne!(ptr, Ptr::null(), "Cannot link null pointer");

        if !prev.is_null() {
            self.links[prev.unchecked_get()].next = ptr;
        }
        if !next.is_null() {
            self.links[next.unchecked_get()].prev = ptr;
        }

        let link = &mut self.links[ptr.unchecked_get()];
        link.prev = prev;
        link.next = next;

        if self.head == next {
            self.head = ptr;
        } else if self.head == ptr {
            self.head = prev;
        }
        if self.tail == prev {
            self.tail = ptr;
        } else if self.tail == ptr {
            self.tail = next;
        }

        Some(())
    }

    pub fn move_to_tail(&mut self, moved: Ptr) {
        self.move_after(moved, self.tail_ptr());
    }

    pub fn move_to_head(&mut self, moved: Ptr) {
        self.move_before(moved, self.head_ptr());
    }

    pub fn ptr_cursor_mut(&'_ mut self, ptr: Ptr) -> CursorMut<'_, K, T> {
        CursorMut { ptr, map: self }
    }

    pub fn next_ptr(&self, ptr: Ptr) -> Option<Ptr> {
        self.links[ptr.get()?].next.optional()
    }

    pub fn prev_ptr(&self, ptr: Ptr) -> Option<Ptr> {
        self.links[ptr.get()?].prev.optional()
    }

    pub fn head_cursor_mut(&'_ mut self) -> CursorMut<'_, K, T> {
        CursorMut {
            ptr: self.head,
            map: self,
        }
    }

    pub fn tail_cursor_mut(&'_ mut self) -> CursorMut<'_, K, T> {
        CursorMut {
            ptr: self.tail,
            map: self,
        }
    }

    pub fn head_ptr(&self) -> Ptr {
        self.head
    }

    pub fn tail_ptr(&self) -> Ptr {
        self.tail
    }

    pub fn swap_remove_ptr(&mut self, ptr: Ptr) -> Option<RemovedEntry<K, T>> {
        if ptr.is_null() {
            return None;
        }

        let invalidated = Ptr::unchecked_from(self.map.len().checked_sub(1)?);

        let (key, value) = self.map.swap_remove_index(ptr.unchecked_get())?;
        let link = self.links.swap_remove(ptr.unchecked_get());

        let next = if link.next == invalidated {
            ptr
        } else {
            link.next
        };
        let prev = if link.prev == invalidated {
            ptr
        } else {
            link.prev
        };

        if ptr.unchecked_get() < self.map.len() {
            let perturbed = &mut self.links[ptr.unchecked_get()];
            let update_prev = perturbed.next;
            if update_prev == ptr {
                perturbed.next = Ptr::null();
            }
            let update_next = perturbed.prev;
            if update_next == ptr {
                perturbed.prev = Ptr::null();
            }

            if update_prev != ptr
                && let Some(update_prev) = self.links.get_mut(update_prev.unchecked_get())
            {
                update_prev.prev = ptr;
            }
            if update_next != ptr
                && let Some(update_next) = self.links.get_mut(update_next.unchecked_get())
            {
                update_next.next = ptr;
            }
        }

        if let Some(update_prev) = self.links.get_mut(next.unchecked_get()) {
            update_prev.prev = prev;
        }
        if let Some(update_next) = self.links.get_mut(prev.unchecked_get()) {
            update_next.next = next;
        }

        if self.tail == ptr {
            self.tail = prev;
        } else if self.tail == invalidated {
            self.tail = ptr;
        }
        if self.head == ptr {
            self.head = next;
        } else if self.head == invalidated {
            self.head = ptr;
        }

        Some(RemovedEntry {
            key,
            value,
            prev,
            next,
            invalidated,
        })
    }

    pub fn ptr_get(&self, ptr: Ptr) -> Option<&T> {
        self.map
            .get_index(ptr.unchecked_get())
            .map(|(_, value)| value)
    }

    pub fn ptr_get_entry(&self, ptr: Ptr) -> Option<(&K, &T)> {
        self.map.get_index(ptr.unchecked_get())
    }

    pub fn ptr_get_entry_mut(&mut self, ptr: Ptr) -> Option<(&K, &mut T)> {
        self.map.get_index_mut(ptr.unchecked_get())
    }

    pub fn ptr_get_mut(&mut self, ptr: Ptr) -> Option<&mut T> {
        self.map
            .get_index_mut(ptr.unchecked_get())
            .map(|(_, value)| value)
    }

    pub fn key_for_ptr(&self, ptr: Ptr) -> Option<&K> {
        self.map.get_index(ptr.unchecked_get()).map(|(k, _)| k)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.links.clear();
        self.head = Ptr::null();
        self.tail = Ptr::null();
    }

    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
    }

    pub fn iter(&'_ self) -> Iter<'_, K, T> {
        Iter {
            ptr: self.head,
            map: self,
        }
    }

    pub fn iter_rev(&'_ self) -> IterRev<'_, K, T> {
        IterRev {
            ptr: self.tail,
            map: self,
        }
    }

    pub fn into_iter(self) -> IntoIter<K, T> {
        IntoIter {
            values: self.map.into_iter().map(Some).collect(),
            links: self.links,
            ptr: self.head,
        }
    }

    pub fn into_iter_rev(self) -> IntoIterRev<K, T> {
        IntoIterRev {
            values: self.map.into_iter().map(Some).collect(),
            links: self.links,
            ptr: self.tail,
        }
    }

    pub fn contains_ptr(&self, ptr: Ptr) -> bool {
        ptr.unchecked_get() < self.map.len()
    }

    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    pub fn debug_validate(&self) {
        if self.is_empty() {
            assert_eq!(self.head, Ptr::null(), "Head pointer should be default");
            assert_eq!(self.tail, Ptr::null(), "Tail pointer should be default");
            return;
        }

        assert_ne!(self.head, Ptr::null(), "Head pointer is invalid");
        assert_ne!(self.tail, Ptr::null(), "Tail pointer is invalid");
        assert_eq!(
            self.prev_ptr(self.head),
            None,
            "Head should not have a previous link"
        );
        assert_eq!(
            self.next_ptr(self.tail),
            None,
            "Tail should not have a next link"
        );
        assert_eq!(
            self.links.len(),
            self.map.len(),
            "Links and map should have the same length"
        );

        for (index, link) in self.links.iter().enumerate() {
            let ptr = Ptr::unchecked_from(index);

            if ptr == self.head {
                assert_eq!(
                    link.prev,
                    Ptr::null(),
                    "Head pointer should not have a previous link"
                );
            } else {
                assert_ne!(
                    link.prev,
                    Ptr::null(),
                    "Non-head pointer should have a previous link: {ptr:?} head: {:?}",
                    self.head
                );
            }

            if ptr == self.tail {
                assert_eq!(
                    link.next,
                    Ptr::null(),
                    "Tail pointer should not have a next link"
                );
            } else {
                assert_ne!(
                    link.next,
                    Ptr::null(),
                    "Non-tail pointer should have a next link: {ptr:?} tail: {:?}",
                    self.tail
                );
            }
        }
    }
}

impl<K: Hash + Eq, T> LinkedHashMap<K, T> {
    pub fn insert_tail(&mut self, key: K, value: T) -> Option<T> {
        let entry = self.entry(key);
        match entry {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                let old = occupied_entry.insert_no_move(value);
                self.move_to_tail(ptr);
                Some(old)
            }
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert_tail(value);
                None
            }
        }
    }

    pub fn insert_head(&mut self, key: K, value: T) -> Option<T> {
        let entry = self.entry(key);
        match entry {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                let old = occupied_entry.insert_no_move(value);
                self.move_to_head(ptr);
                Some(old)
            }
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert_head(value);
                None
            }
        }
    }

    pub fn key_cursor_mut(&'_ mut self, key: &K) -> CursorMut<'_, K, T> {
        let ptr = self
            .map
            .get_index_of(key)
            .map(Ptr::unchecked_from)
            .unwrap_or_default();
        CursorMut { ptr, map: self }
    }

    pub fn entry(&'_ mut self, key: K) -> Entry<'_, K, T> {
        match self.map.entry(key) {
            map::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry { entry }),
            map::Entry::Vacant(entry) => Entry::Vacant(VacantEntry {
                entry,
                head: &mut self.head,
                tail: &mut self.tail,
                links: &mut self.links,
            }),
        }
    }

    pub fn swap_remove(&mut self, key: &K) -> Option<RemovedEntry<K, T>> {
        let ptr = Ptr::unchecked_from(self.map.get_index_of(key)?);
        self.swap_remove_ptr(ptr)
    }

    pub fn get_ptr(&self, key: &K) -> Option<Ptr> {
        self.map.get_index_of(key).map(Ptr::unchecked_from)
    }

    pub fn get(&self, key: &K) -> Option<&T> {
        self.map.get(key)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut T> {
        self.map.get_mut(key)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct CursorMut<'m, K, T> {
    ptr: Ptr,
    map: &'m mut LinkedHashMap<K, T>,
}

impl<'m, K: Hash + Eq, T> CursorMut<'m, K, T> {
    pub fn insert_after_move_to(&mut self, key: K, value: T) -> Option<T> {
        match self.map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                self.map.move_after(ptr, self.ptr);
                self.ptr = ptr;
                Some(std::mem::replace(self.map.ptr_get_mut(ptr)?, value))
            }
            Entry::Vacant(vacant_entry) => {
                self.ptr = vacant_entry.insert_after(value, self.ptr);
                None
            }
        }
    }

    pub fn insert_before_move_to(&mut self, key: K, value: T) -> Option<T> {
        match self.map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                self.map.move_before(ptr, self.ptr);
                self.ptr = ptr;
                Some(std::mem::replace(self.map.ptr_get_mut(ptr)?, value))
            }
            Entry::Vacant(vacant_entry) => {
                self.ptr = vacant_entry.insert_before(value, self.ptr);
                None
            }
        }
    }

    pub fn iter(&self) -> Iter<'_, K, T> {
        Iter {
            ptr: self.ptr,
            map: self.map,
        }
    }

    pub fn iter_rev(&self) -> IterRev<'_, K, T> {
        IterRev {
            ptr: self.ptr,
            map: self.map,
        }
    }

    pub fn get_ptr(&self, key: &K) -> Option<Ptr> {
        self.map.get_ptr(key)
    }

    pub fn remove_prev(&mut self) -> Option<RemovedEntry<K, T>> {
        self.map.swap_remove_ptr(self.map.prev_ptr(self.ptr)?)
    }

    pub fn remove_next(&mut self) -> Option<RemovedEntry<K, T>> {
        self.map.swap_remove_ptr(self.map.next_ptr(self.ptr)?)
    }

    pub fn remove(self) -> Option<RemovedEntry<K, T>> {
        self.map.swap_remove_ptr(self.ptr)
    }

    pub fn move_next(&mut self) {
        self.ptr = self.map.next_ptr(self.ptr).unwrap_or_default();
    }

    pub fn move_prev(&mut self) {
        self.ptr = self.map.prev_ptr(self.ptr).unwrap_or_default();
    }

    pub fn ptr(&self) -> Ptr {
        self.ptr
    }

    pub fn current(&self) -> Option<(&K, &T)> {
        self.map.map.get_index(self.ptr.unchecked_get())
    }

    pub fn current_mut(&mut self) -> Option<(&K, &mut T)> {
        self.map.map.get_index_mut(self.ptr.unchecked_get())
    }

    pub fn next_ptr(&self) -> Option<Ptr> {
        self.map.next_ptr(self.ptr)
    }

    pub fn next(&self) -> Option<(&K, &T)> {
        let ptr = self.next_ptr()?;
        self.map.map.get_index(ptr.unchecked_get())
    }

    pub fn next_mut(&mut self) -> Option<(&K, &mut T)> {
        let ptr = self.next_ptr()?;
        self.map.map.get_index_mut(ptr.unchecked_get())
    }

    pub fn prev_ptr(&self) -> Option<Ptr> {
        self.map.prev_ptr(self.ptr)
    }

    pub fn prev(&self) -> Option<(&K, &T)> {
        let ptr = self.prev_ptr()?;
        self.map.map.get_index(ptr.unchecked_get())
    }

    pub fn prev_mut(&mut self) -> Option<(&K, &mut T)> {
        let ptr = self.prev_ptr()?;
        self.map.map.get_index_mut(ptr.unchecked_get())
    }
}

impl<K, T> Index<Ptr> for LinkedHashMap<K, T> {
    type Output = T;

    fn index(&self, index: Ptr) -> &Self::Output {
        &self.map[index.unchecked_get()]
    }
}

impl<K, T> IndexMut<Ptr> for LinkedHashMap<K, T> {
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        &mut self.map[index.unchecked_get()]
    }
}

#[doc(hidden)]
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

#[doc(hidden)]
pub struct OccupiedEntry<'a, K, V> {
    entry: map::OccupiedEntry<'a, K, V>,
}

impl<K, V> OccupiedEntry<'_, K, V> {
    pub fn get(&self) -> &V {
        self.entry.get()
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.entry.get_mut()
    }

    pub fn insert_no_move(mut self, value: V) -> V {
        std::mem::replace(self.entry.get_mut(), value)
    }

    pub fn ptr(&self) -> Ptr {
        Ptr::unchecked_from(self.entry.index())
    }

    pub fn key(&self) -> &K {
        self.entry.key()
    }
}

#[doc(hidden)]
pub struct VacantEntry<'a, K, V> {
    entry: map::VacantEntry<'a, K, V>,
    links: &'a mut Vec<Link>,
    head: &'a mut Ptr,
    tail: &'a mut Ptr,
}

impl<K: Hash + Eq, V> VacantEntry<'_, K, V> {
    pub fn entry_ptr_unstable(&self) -> Ptr {
        Ptr::unchecked_from(self.entry.index())
    }

    pub fn insert_tail(self, value: V) -> Ptr {
        let after = *self.tail;
        self.insert_after(value, after)
    }

    pub fn push_unlinked(self, value: V) {
        let ptr = Ptr::unchecked_from(self.entry.index());
        if self.head.is_null() && self.tail.is_null() {
            *self.head = ptr;
            *self.tail = ptr;
        }

        self.entry.insert(value);
        self.links.push(Link {
            prev: Ptr::null(),
            next: Ptr::null(),
        });
    }

    pub fn insert_after(self, value: V, after: Ptr) -> Ptr {
        let ptr = Ptr::unchecked_from(self.entry.index());
        if self.head.is_null() && self.tail.is_null() {
            *self.head = ptr;
            *self.tail = ptr;
            self.entry.insert(value);
            self.links.push(Link {
                prev: Ptr::null(),
                next: Ptr::null(),
            });
        } else {
            let after_next = self
                .links
                .get(after.unchecked_get())
                .map_or(Ptr::null(), |l| l.next);
            self.entry.insert(value);
            self.links.push(Link {
                prev: after,
                next: after_next,
            });

            if let Some(p) = self.links.get_mut(after.unchecked_get()) {
                p.next = ptr;
            }
            if let Some(p) = self.links.get_mut(after_next.unchecked_get()) {
                p.prev = ptr;
            }

            if *self.tail == after {
                *self.tail = ptr;
            }
        }
        ptr
    }

    pub fn insert_head(self, value: V) -> Ptr {
        let ptr = *self.head;
        self.insert_before(value, ptr)
    }

    pub fn insert_before(self, value: V, before: Ptr) -> Ptr {
        let ptr = Ptr::unchecked_from(self.entry.index());
        if self.head.is_null() && self.tail.is_null() {
            *self.head = ptr;
            *self.tail = ptr;
            self.entry.insert(value);
            self.links.push(Link {
                prev: Ptr::null(),
                next: Ptr::null(),
            });
        } else {
            let before_prev = self
                .links
                .get(before.unchecked_get())
                .map_or(Ptr::null(), |l| l.prev);
            self.entry.insert(value);
            self.links.push(Link {
                prev: before_prev,
                next: before,
            });

            if let Some(p) = self.links.get_mut(before.unchecked_get()) {
                p.prev = ptr;
            }
            if let Some(p) = self.links.get_mut(before_prev.unchecked_get()) {
                p.next = ptr;
            }

            if *self.head == before {
                *self.head = ptr;
            }
        }
        ptr
    }

    pub fn into_key(self) -> K {
        self.entry.into_key()
    }

    pub fn key(&self) -> &K {
        self.entry.key()
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct Iter<'a, K, T> {
    ptr: Ptr,
    map: &'a LinkedHashMap<K, T>,
}

impl<'a, K, T> Iterator for Iter<'a, K, T> {
    type Item = (&'a K, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.ptr;
        self.ptr = self.map.next_ptr(ptr).unwrap_or_default();
        self.map.ptr_get_entry(ptr)
    }
}

#[doc(hidden)]
pub struct IterRev<'a, K, T> {
    ptr: Ptr,
    map: &'a LinkedHashMap<K, T>,
}

impl<'a, K, T> Iterator for IterRev<'a, K, T> {
    type Item = (&'a K, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.ptr;
        self.ptr = self.map.prev_ptr(ptr).unwrap_or_default();
        self.map.ptr_get_entry(ptr)
    }
}

#[doc(hidden)]
pub struct IntoIter<K, T> {
    values: Vec<Option<(K, T)>>,
    ptr: Ptr,
    links: Vec<Link>,
}

impl<K, T> Iterator for IntoIter<K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            return None;
        }

        let ptr = self.ptr;
        self.ptr = self.links[ptr.unchecked_get()].next;
        self.values[ptr.unchecked_get()].take()
    }
}

#[doc(hidden)]
pub struct IntoIterRev<K, T> {
    values: Vec<Option<(K, T)>>,
    ptr: Ptr,
    links: Vec<Link>,
}

impl<K, T> Iterator for IntoIterRev<K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            return None;
        }

        let ptr = self.ptr;

        self.ptr = self.links[ptr.unchecked_get()].prev;
        self.values[ptr.unchecked_get()].take()
    }
}

#[cfg(test)]
mod tests {
    use ntest::timeout;

    use super::*;

    #[test]
    #[timeout(1000)]
    fn test_new_and_default() {
        let map: LinkedHashMap<i32, String> = LinkedHashMap::default();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.head_ptr(), Ptr::null());
        assert_eq!(map.tail_ptr(), Ptr::null());
    }

    #[test]
    #[timeout(1000)]
    fn test_with_capacity() {
        let map: LinkedHashMap<i32, String> = LinkedHashMap::with_capacity(10);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    #[timeout(1000)]
    fn test_clear() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());

        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());

        map.clear();

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.head_ptr(), Ptr::null());
        assert_eq!(map.tail_ptr(), Ptr::null());
    }

    #[test]
    #[timeout(1000)]
    fn test_insert_tail() {
        let mut map = LinkedHashMap::default();

        let result = map.insert_tail(1, "one".to_string());
        assert_eq!(result, None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one".to_string()));
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let result = map.insert_tail(2, "two".to_string());
        assert_eq!(result, None);
        assert_eq!(map.len(), 2);
        assert_ne!(map.head_ptr(), map.tail_ptr());

        map.insert_tail(3, "three".to_string());
        assert_eq!(map.len(), 3);

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&1, &"one".to_string()),
                (&2, &"two".to_string()),
                (&3, &"three".to_string())
            ]
        );

        let result = map.insert_tail(2, "TWO".to_string());
        assert_eq!(result, Some("two".to_string()));
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&"TWO".to_string()));

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&1, &"one".to_string()),
                (&3, &"three".to_string()),
                (&2, &"TWO".to_string())
            ]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_insert_head() {
        let mut map = LinkedHashMap::default();

        let result = map.insert_head(1, "one".to_string());
        assert_eq!(result, None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one".to_string()));
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let result = map.insert_head(2, "two".to_string());
        assert_eq!(result, None);
        assert_eq!(map.len(), 2);
        assert_ne!(map.head_ptr(), map.tail_ptr());

        map.insert_head(3, "three".to_string());
        assert_eq!(map.len(), 3);

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&3, &"three".to_string()),
                (&2, &"two".to_string()),
                (&1, &"one".to_string())
            ]
        );

        let result = map.insert_head(2, "TWO".to_string());
        assert_eq!(result, Some("two".to_string()));
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&"TWO".to_string()));

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&2, &"TWO".to_string()),
                (&3, &"three".to_string()),
                (&1, &"one".to_string())
            ]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_mixed_insertion() {
        let mut map = LinkedHashMap::default();

        map.insert_tail(1, "one");
        map.insert_head(2, "two");
        map.insert_tail(3, "three");
        map.insert_head(4, "four");

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![(&4, &"four"), (&2, &"two"), (&1, &"one"), (&3, &"three")]
        );
        assert_eq!(map.len(), 4);
    }

    #[test]
    #[timeout(1000)]
    fn test_get_operations() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());
        map.insert_tail(3, "three".to_string());

        assert_eq!(map.get(&1), Some(&"one".to_string()));
        assert_eq!(map.get(&2), Some(&"two".to_string()));
        assert_eq!(map.get(&3), Some(&"three".to_string()));
        assert_eq!(map.get(&4), None);

        let value = map.get_mut(&2).unwrap();
        *value = "TWO".to_string();
        assert_eq!(map.get(&2), Some(&"TWO".to_string()));

        assert!(map.contains_key(&1));
        assert!(map.contains_key(&2));
        assert!(map.contains_key(&3));
        assert!(!map.contains_key(&4));
        assert!(!map.contains_key(&0));
    }

    #[test]
    #[timeout(1000)]
    fn test_get_ptr_operations() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr2 = map.get_ptr(&2).unwrap();
        assert_ne!(ptr1, ptr2);
        assert_eq!(map.get_ptr(&99), None);

        assert_eq!(map.ptr_get(ptr1), Some(&"one".to_string()));
        assert_eq!(map.ptr_get(ptr2), Some(&"two".to_string()));
        assert_eq!(map.ptr_get(Ptr::null()), None);

        let value = map.ptr_get_mut(ptr1).unwrap();
        *value = "ONE".to_string();
        assert_eq!(map.ptr_get(ptr1), Some(&"ONE".to_string()));

        let (key, value) = map.ptr_get_entry(ptr1).unwrap();
        assert_eq!(key, &1);
        assert_eq!(value, &"ONE".to_string());

        let (key, value) = map.ptr_get_entry_mut(ptr2).unwrap();
        assert_eq!(key, &2);
        *value = "TWO".to_string();
        assert_eq!(map.get(&2), Some(&"TWO".to_string()));

        assert_eq!(map.key_for_ptr(ptr1), Some(&1));
        assert_eq!(map.key_for_ptr(ptr2), Some(&2));
        assert_eq!(map.key_for_ptr(Ptr::null()), None);

        assert!(map.contains_ptr(ptr1));
        assert!(map.contains_ptr(ptr2));
        assert!(!map.contains_ptr(Ptr::null()));
    }

    #[test]
    #[timeout(1000)]
    fn test_index_operations() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr2 = map.get_ptr(&2).unwrap();

        assert_eq!(&map[ptr1], &"one".to_string());
        assert_eq!(&map[ptr2], &"two".to_string());

        map[ptr1] = "ONE".to_string();
        assert_eq!(&map[ptr1], &"ONE".to_string());
        assert_eq!(map.get(&1), Some(&"ONE".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_ptr_index_mapping() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(10, "ten");
        map.insert_tail(20, "twenty");
        map.insert_tail(30, "thirty");

        let ptr0 = map.index_ptr_unstable(0).unwrap();
        let ptr1 = map.index_ptr_unstable(1).unwrap();
        let ptr2 = map.index_ptr_unstable(2).unwrap();

        assert_eq!(map.ptr_index_unstable(ptr0), Some(0));
        assert_eq!(map.ptr_index_unstable(ptr1), Some(1));
        assert_eq!(map.ptr_index_unstable(ptr2), Some(2));

        assert_eq!(map.ptr_get(ptr0), Some(&"ten"));
        assert_eq!(map.ptr_get(ptr1), Some(&"twenty"));
        assert_eq!(map.ptr_get(ptr2), Some(&"thirty"));
    }

    #[test]
    #[timeout(1000)]
    fn test_remove_by_key() {
        let mut map: LinkedHashMap<i32, String> = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());
        map.insert_tail(3, "three".to_string());

        let removed: Option<RemovedEntry<i32, String>> = map.swap_remove(&2);
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 2,
                value: "two".to_string(),
                prev: map.get_ptr(&1).unwrap(),
                next: map.get_ptr(&3).unwrap(),
                invalidated: Ptr::unchecked_from(2),
            })
        );
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key(&2));

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![(&1, &"one".to_string()), (&3, &"three".to_string())]
        );

        let removed = map.swap_remove(&1);
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 1,
                value: "one".to_string(),
                prev: Ptr::null(),
                next: map.get_ptr(&3).unwrap(),
                invalidated: Ptr::unchecked_from(1),
            })
        );
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let removed = map.swap_remove(&3);
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 3,
                value: "three".to_string(),
                prev: Ptr::null(),
                next: Ptr::null(),
                invalidated: Ptr::unchecked_from(0),
            })
        );
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.head_ptr(), Ptr::null());
        assert_eq!(map.tail_ptr(), Ptr::null());

        let removed = map.swap_remove(&1);
        assert_eq!(removed, None);
    }

    #[test]
    #[timeout(1000)]
    fn test_remove_by_ptr() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());
        map.insert_tail(3, "three".to_string());

        let removed = map.swap_remove_ptr(map.get_ptr(&2).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 2,
                value: "two".to_string(),
                prev: map.get_ptr(&1).unwrap(),
                next: map.get_ptr(&3).unwrap(),
                invalidated: Ptr::unchecked_from(2),
            })
        );
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key(&2));

        let removed = map.swap_remove_ptr(map.get_ptr(&1).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 1,
                value: "one".to_string(),
                prev: Ptr::null(),
                next: map.get_ptr(&3).unwrap(),
                invalidated: Ptr::unchecked_from(1),
            })
        );
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.get_ptr(&3).unwrap());
        assert_eq!(map.tail_ptr(), map.get_ptr(&3).unwrap());

        let removed = map.swap_remove_ptr(map.get_ptr(&1).unwrap_or(Ptr::null()));
        assert_eq!(removed, None);

        let removed = map.swap_remove_ptr(map.get_ptr(&3).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 3,
                value: "three".to_string(),
                prev: Ptr::null(),
                next: Ptr::null(),
                invalidated: Ptr::unchecked_from(0),
            })
        );
        assert!(map.is_empty());
    }

    #[test]
    #[timeout(1000)]
    fn test_remove_single_element() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(42, "answer".to_string());

        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let removed = map.swap_remove(&42);
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 42,
                value: "answer".to_string(),
                prev: Ptr::null(),
                next: Ptr::null(),
                invalidated: Ptr::unchecked_from(0),
            })
        );
        assert!(map.is_empty());
        assert_eq!(map.head_ptr(), Ptr::null());
        assert_eq!(map.tail_ptr(), Ptr::null());
    }

    #[test]
    #[timeout(1000)]
    fn test_remove_head_and_tail() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        let removed = map.swap_remove(&1);
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 1,
                value: "value1".to_string(),
                prev: Ptr::null(),
                next: map.get_ptr(&2).unwrap(),
                invalidated: Ptr::unchecked_from(4),
            })
        );
        assert_eq!(map.tail_ptr(), map.get_ptr(&5).unwrap());

        let removed = map.swap_remove(&5);
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 5,
                value: "value5".to_string(),
                prev: map.get_ptr(&4).unwrap(),
                next: Ptr::null(),
                invalidated: Ptr::unchecked_from(3),
            })
        );
        assert_eq!(map.len(), 3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 3, 4]);
    }

    #[test]
    #[timeout(1000)]
    fn test_move_to_head() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr3 = map.get_ptr(&3).unwrap();

        map.move_to_head(ptr3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![3, 1, 2, 4]);
        assert_eq!(map.head_ptr(), ptr3);

        let old_head = map.head_ptr();
        map.move_to_head(old_head);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![3, 1, 2, 4]);

        let ptr4 = map.get_ptr(&4).unwrap();
        map.move_to_head(ptr4);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![4, 3, 1, 2]);
        assert_eq!(map.head_ptr(), ptr4);
    }

    #[test]
    #[timeout(1000)]
    fn test_move_to_tail() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr2 = map.get_ptr(&2).unwrap();

        map.move_to_tail(ptr2);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 3, 4, 2]);
        assert_eq!(map.tail_ptr(), ptr2);

        let old_tail = map.tail_ptr();
        map.move_to_tail(old_tail);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 3, 4, 2]);

        let ptr1 = map.get_ptr(&1).unwrap();
        map.move_to_tail(ptr1);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![3, 4, 2, 1]);
        assert_eq!(map.tail_ptr(), ptr1);
    }

    #[test]
    #[timeout(1000)]
    fn test_move_after() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr3 = map.get_ptr(&3).unwrap();
        let ptr5 = map.get_ptr(&5).unwrap();

        map.move_after(ptr5, ptr1);
        assert_eq!(map.next_ptr(ptr1), Some(ptr5));
        assert_eq!(map.prev_ptr(ptr5), Some(ptr1));
        assert_eq!(map.next_ptr(ptr5), map.get_ptr(&2));
        assert_eq!(map.prev_ptr(map.get_ptr(&2).unwrap()), Some(ptr5));
        assert_eq!(map.next_ptr(ptr3), map.get_ptr(&4));
        assert_eq!(map.prev_ptr(map.get_ptr(&4).unwrap()), Some(ptr3));

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5, 2, 3, 4]);

        let ptr2 = map.get_ptr(&2).unwrap();
        let ptr4 = map.get_ptr(&4).unwrap();
        map.move_after(ptr2, ptr4);
        assert_eq!(map.next_ptr(ptr4), Some(ptr2));
        assert_eq!(map.prev_ptr(ptr2), Some(ptr4));
        assert_eq!(map.next_ptr(ptr5), Some(ptr3));
        assert_eq!(map.prev_ptr(ptr3), Some(ptr5));

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5, 3, 4, 2]);

        map.move_after(ptr3, ptr3);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5, 3, 4, 2]);

        map.move_after(ptr4, ptr3);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5, 3, 4, 2]);
    }

    #[test]
    #[timeout(1000)]
    fn test_move_before() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr3 = map.get_ptr(&3).unwrap();
        let ptr5 = map.get_ptr(&5).unwrap();

        map.move_before(ptr5, ptr3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 2, 5, 3, 4]);

        let ptr4 = map.get_ptr(&4).unwrap();
        map.move_before(ptr1, ptr4);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 5, 3, 1, 4]);

        map.move_before(ptr3, ptr3);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 5, 3, 1, 4]);

        let ptr2 = map.get_ptr(&2).unwrap();
        map.move_before(ptr2, ptr5);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 5, 3, 1, 4]);
    }

    #[test]
    #[timeout(1000)]
    fn test_pointer_navigation() {
        let mut map = LinkedHashMap::default();
        for i in 1..=3 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr2 = map.get_ptr(&2).unwrap();
        let ptr3 = map.get_ptr(&3).unwrap();

        assert_eq!(map.next_ptr(ptr1), Some(ptr2));
        assert_eq!(map.next_ptr(ptr2), Some(ptr3));
        assert_eq!(map.next_ptr(ptr3), None);
        assert_eq!(map.next_ptr(Ptr::null()), None);

        assert_eq!(map.prev_ptr(ptr1), None);
        assert_eq!(map.prev_ptr(ptr2), Some(ptr1));
        assert_eq!(map.prev_ptr(ptr3), Some(ptr2));
        assert_eq!(map.prev_ptr(Ptr::null()), None);
    }

    #[test]
    #[timeout(1000)]
    fn test_move_operations_edge_cases() {
        let mut map = LinkedHashMap::default();

        map.move_to_head(Ptr::null());
        map.move_to_tail(Ptr::null());
        map.move_after(Ptr::null(), Ptr::null());
        map.move_before(Ptr::null(), Ptr::null());
        assert!(map.is_empty());

        map.insert_tail(1, "one");
        let ptr1 = map.get_ptr(&1).unwrap();

        map.move_to_head(ptr1);
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), ptr1);
        assert_eq!(map.tail_ptr(), ptr1);

        map.move_to_tail(ptr1);
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), ptr1);
        assert_eq!(map.tail_ptr(), ptr1);
    }

    #[test]
    #[timeout(1000)]
    fn test_iter() {
        let mut map = LinkedHashMap::default();

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![]);

        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&1, &"value1".to_string()),
                (&2, &"value2".to_string()),
                (&3, &"value3".to_string()),
                (&4, &"value4".to_string())
            ]
        );

        map.insert_head(0, "value0".to_string());
        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&0, &"value0".to_string()),
                (&1, &"value1".to_string()),
                (&2, &"value2".to_string()),
                (&3, &"value3".to_string()),
                (&4, &"value4".to_string())
            ]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_iter_rev() {
        let mut map = LinkedHashMap::default();

        let items: Vec<_> = map.iter_rev().collect();
        assert_eq!(items, vec![]);

        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let items: Vec<_> = map.iter_rev().collect();
        assert_eq!(
            items,
            vec![
                (&4, &"value4".to_string()),
                (&3, &"value3".to_string()),
                (&2, &"value2".to_string()),
                (&1, &"value1".to_string())
            ]
        );

        map.insert_head(0, "value0".to_string());
        let items: Vec<_> = map.iter_rev().collect();
        assert_eq!(
            items,
            vec![
                (&4, &"value4".to_string()),
                (&3, &"value3".to_string()),
                (&2, &"value2".to_string()),
                (&1, &"value1".to_string()),
                (&0, &"value0".to_string())
            ]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_into_iter() {
        let mut map = LinkedHashMap::default();

        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let items: Vec<_> = map.into_iter().collect();
        assert_eq!(
            items,
            vec![
                (1, "value1".to_string()),
                (2, "value2".to_string()),
                (3, "value3".to_string()),
                (4, "value4".to_string())
            ]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_into_iter_rev() {
        let mut map = LinkedHashMap::default();

        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let items: Vec<_> = map.into_iter_rev().collect();
        assert_eq!(
            items,
            vec![
                (4, "value4".to_string()),
                (3, "value3".to_string()),
                (2, "value2".to_string()),
                (1, "value1".to_string())
            ]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_iteration_after_modifications() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        map.swap_remove(&3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 2, 4, 5]);

        let ptr2 = map.get_ptr(&2).unwrap();
        map.move_to_tail(ptr2);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 4, 5, 2]);

        let items: Vec<_> = map.iter_rev().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 5, 4, 1]);
    }

    #[test]
    #[timeout(1000)]
    fn test_empty_iteration() {
        let map: LinkedHashMap<i32, String> = LinkedHashMap::default();

        assert_eq!(map.iter().count(), 0);
        assert_eq!(map.iter_rev().count(), 0);

        let empty_map: LinkedHashMap<i32, String> = LinkedHashMap::default();
        assert_eq!(empty_map.into_iter().count(), 0);

        let empty_map: LinkedHashMap<i32, String> = LinkedHashMap::default();
        assert_eq!(empty_map.into_iter_rev().count(), 0);
    }

    #[test]
    #[timeout(1000)]
    fn test_single_element_iteration() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(42, "answer".to_string());

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&42, &"answer".to_string())]);

        let items: Vec<_> = map.iter_rev().collect();
        assert_eq!(items, vec![(&42, &"answer".to_string())]);
    }

    #[test]
    #[timeout(1000)]
    fn test_entry_api_vacant() {
        let mut map = LinkedHashMap::default();

        match map.entry(1) {
            Entry::Vacant(entry) => {
                assert_eq!(entry.key(), &1);
                let ptr = entry.insert_tail("one".to_string());
                assert_ne!(ptr, Ptr::null());
            }
            Entry::Occupied(_) => panic!("Expected vacant entry"),
        }

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one".to_string()));

        match map.entry(2) {
            Entry::Vacant(entry) => {
                let key = entry.into_key();
                assert_eq!(key, 2);
            }
            Entry::Occupied(_) => panic!("Expected vacant entry"),
        }

        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&2));
    }

    #[test]
    #[timeout(1000)]
    fn test_entry_api_occupied() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());

        match map.entry(1) {
            Entry::Occupied(entry) => {
                assert_eq!(entry.key(), &1);
                assert_eq!(entry.get(), &"one".to_string());
                assert_ne!(entry.ptr(), Ptr::null());
            }
            Entry::Vacant(_) => panic!("Expected occupied entry"),
        }

        match map.entry(2) {
            Entry::Occupied(mut entry) => {
                let value = entry.get_mut();
                *value = "TWO".to_string();
            }
            Entry::Vacant(_) => panic!("Expected occupied entry"),
        }

        assert_eq!(map.get(&2), Some(&"TWO".to_string()));

        match map.entry(1) {
            Entry::Occupied(entry) => {
                let old_value = entry.insert_no_move("ONE".to_string());
                assert_eq!(old_value, "one".to_string());
            }
            Entry::Vacant(_) => panic!("Expected occupied entry"),
        }

        assert_eq!(map.get(&1), Some(&"ONE".to_string()));
        assert_eq!(map.len(), 2);
    }

    #[test]
    #[timeout(1000)]
    fn test_entry_api_mixed_operations() {
        let mut map = LinkedHashMap::default();

        for i in 1..=3 {
            match map.entry(i) {
                Entry::Vacant(entry) => {
                    entry.insert_tail(format!("value{}", i));
                }
                Entry::Occupied(_) => panic!("Unexpected occupied entry"),
            }
        }

        assert_eq!(map.len(), 3);

        match map.entry(2) {
            Entry::Occupied(entry) => {
                entry.insert_no_move("updated".to_string());
            }
            Entry::Vacant(_) => panic!("Expected occupied entry"),
        }

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&1, &"value1".to_string()),
                (&2, &"updated".to_string()),
                (&3, &"value3".to_string())
            ]
        );
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_basic_operations() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let mut cursor = map.head_cursor_mut();
        assert_eq!(cursor.current(), Some((&1, &"value1".to_string())));

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&2, &"value2".to_string())));

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&3, &"value3".to_string())));

        cursor.move_prev();
        assert_eq!(cursor.current(), Some((&2, &"value2".to_string())));

        let mut cursor = map.tail_cursor_mut();
        assert_eq!(cursor.current(), Some((&4, &"value4".to_string())));

        cursor.move_prev();
        assert_eq!(cursor.current(), Some((&3, &"value3".to_string())));
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_current_operations() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());

        let mut cursor = map.key_cursor_mut(&1);

        if let Some((key, value)) = cursor.current_mut() {
            assert_eq!(key, &1);
            *value = "ONE".to_string();
        }

        assert_eq!(map.get(&1), Some(&"ONE".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_next_prev_operations() {
        let mut map = LinkedHashMap::default();
        for i in 1..=3 {
            map.insert_tail(i, format!("value{}", i));
        }

        let mut cursor = map.head_cursor_mut();

        assert_eq!(cursor.next(), Some((&2, &"value2".to_string())));

        if let Some((key, value)) = cursor.next_mut() {
            assert_eq!(key, &2);
            *value = "VALUE2".to_string();
        }

        cursor.move_next();
        cursor.move_next();

        assert_eq!(cursor.prev(), Some((&2, &"VALUE2".to_string())));

        if let Some((key, value)) = cursor.prev_mut() {
            assert_eq!(key, &2);
            *value = "value2_updated".to_string();
        }

        assert_eq!(map.get(&2), Some(&"value2_updated".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_insert_after_move_to() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(3, "three".to_string());

        let mut cursor = map.key_cursor_mut(&1);

        let old_value = cursor.insert_after_move_to(2, "two".to_string());
        assert_eq!(old_value, None);

        let items: Vec<_> = cursor.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 3]);

        let old_value = cursor.insert_after_move_to(2, "TWO".to_string());
        assert_eq!(old_value, Some("two".to_string()));

        assert_eq!(map.get(&2), Some(&"TWO".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_insert_before_move_to() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(3, "three".to_string());

        let mut cursor = map.key_cursor_mut(&3);

        let old_value = cursor.insert_before_move_to(2, "two".to_string());
        assert_eq!(old_value, None);

        let items: Vec<_> = cursor.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 3]);

        let old_value = cursor.insert_before_move_to(2, "TWO".to_string());
        assert_eq!(old_value, Some("two".to_string()));

        assert_eq!(map.get(&2), Some(&"TWO".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_remove_operations() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        let mut cursor = map.key_cursor_mut(&3);

        let removed = cursor.remove_next();
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 4,
                value: "value4".to_string(),
                prev: cursor.get_ptr(&3).unwrap(),
                next: cursor.get_ptr(&5).unwrap(),
                invalidated: Ptr::unchecked_from(4),
            })
        );

        let removed = cursor.remove_prev();
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 2,
                value: "value2".to_string(),
                prev: cursor.get_ptr(&1).unwrap(),
                next: cursor.get_ptr(&3).unwrap(),
                invalidated: Ptr::unchecked_from(3),
            })
        );

        let removed = cursor.remove();
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 3,
                value: "value3".to_string(),
                prev: map.get_ptr(&1).unwrap(),
                next: map.get_ptr(&5).unwrap(),
                invalidated: Ptr::unchecked_from(2),
            })
        );
        assert!(!map.contains_key(&3));

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5]);
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_remove_entry() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());

        let cursor = map.key_cursor_mut(&1);
        let removed = cursor.remove();
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 1,
                value: "one".to_string(),
                prev: Ptr::null(),
                next: map.get_ptr(&2).unwrap(),
                invalidated: Ptr::unchecked_from(1),
            })
        );

        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&1));
        assert_eq!(map.get(&2), Some(&"two".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_empty_map() {
        let mut map: LinkedHashMap<i32, String> = LinkedHashMap::default();

        let mut cursor = map.head_cursor_mut();
        assert_eq!(cursor.current(), None);
        assert_eq!(cursor.next(), None);
        assert_eq!(cursor.prev(), None);
        assert_eq!(cursor.next_ptr(), None);
        assert_eq!(cursor.prev_ptr(), None);

        let old_value = cursor.insert_after_move_to(1, "one".to_string());
        assert_eq!(old_value, None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_edge_cases_invalid_pointers() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());

        let invalid_ptr = Ptr::null();

        assert_eq!(map.ptr_get(invalid_ptr), None);
        assert_eq!(map.ptr_get_entry(invalid_ptr), None);
        assert_eq!(map.ptr_get_entry_mut(invalid_ptr), None);
        assert_eq!(map.ptr_get_mut(invalid_ptr), None);
        assert_eq!(map.key_for_ptr(invalid_ptr), None);
        assert_eq!(map.swap_remove_ptr(invalid_ptr), None);
        assert!(!map.contains_ptr(invalid_ptr));

        map.move_to_head(invalid_ptr);
        map.move_to_tail(invalid_ptr);
        map.move_after(invalid_ptr, invalid_ptr);
        map.move_before(invalid_ptr, invalid_ptr);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_stress_large_operations() {
        let mut map = LinkedHashMap::default();
        const SIZE: usize = 1000;

        for i in 0..SIZE {
            map.insert_tail(i, format!("value{}", i));
        }

        assert_eq!(map.len(), SIZE);

        for i in 0..SIZE {
            assert_eq!(map.get(&i), Some(&format!("value{}", i)));
        }

        for i in (0..SIZE).step_by(10) {
            if let Some(ptr) = map.get_ptr(&i) {
                map.move_to_head(ptr);
            }
        }

        assert_eq!(map.len(), SIZE);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items.len(), SIZE);

        for i in (0..SIZE).step_by(2) {
            map.swap_remove(&i);
        }

        assert_eq!(map.len(), SIZE / 2);

        for i in (1..SIZE).step_by(2) {
            assert_eq!(map.get(&i), Some(&format!("value{}", i)));
        }
    }

    #[test]
    #[timeout(1000)]
    fn test_complex_movement_patterns() {
        let mut map = LinkedHashMap::default();
        for i in 1..=10 {
            map.insert_tail(i, i);
        }

        let ptr5 = map.get_ptr(&5).unwrap();
        let ptr2 = map.get_ptr(&2).unwrap();
        let ptr8 = map.get_ptr(&8).unwrap();

        map.move_after(ptr5, ptr8);

        map.move_before(ptr2, ptr5);

        map.move_to_head(ptr8);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items[0], 8);
        assert!(items.contains(&2));
        assert!(items.contains(&5));
        assert_eq!(map.len(), 10);

        for i in 1..=10 {
            assert!(map.contains_key(&i));
        }
    }

    #[test]
    #[timeout(1000)]
    fn test_iteration_consistency_after_modifications() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, i * 10);
        }

        map.swap_remove(&3);
        if let Some(ptr) = map.get_ptr(&1) {
            map.move_to_tail(ptr);
        }
        map.insert_head(0, 0);

        let forward: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        let backward: Vec<_> = map.iter_rev().map(|(k, _)| *k).collect();
        let mut backward_rev = backward.clone();
        backward_rev.reverse();

        assert_eq!(forward, backward_rev);

        let map_clone = map.clone();
        let consumed: Vec<_> = map_clone.into_iter().map(|(k, _)| k).collect();
        assert_eq!(forward, consumed);

        let consumed_rev: Vec<_> = map.into_iter_rev().map(|(k, _)| k).collect();
        assert_eq!(backward, consumed_rev);
    }

    #[test]
    #[timeout(1000)]
    fn test_shrink_to_fit() {
        let mut map = LinkedHashMap::with_capacity(100);

        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        map.shrink_to_fit();

        assert_eq!(map.len(), 5);
        for i in 1..=5 {
            assert_eq!(map.get(&i), Some(&format!("value{}", i)));
        }

        map.insert_tail(6, "value6".to_string());
        assert_eq!(map.len(), 6);
    }

    #[cfg(all(debug_assertions, feature = "internal-debugging"))]
    #[test]
    #[timeout(1000)]
    fn test_debug_validation() {
        let mut map: LinkedHashMap<i32, String> = LinkedHashMap::default();

        map.debug_validate();

        map.insert_tail(1, "one".to_string());
        map.debug_validate();

        for i in 2..=5 {
            map.insert_tail(i, format!("value{}", i));
            map.debug_validate();
        }

        if let Some(ptr) = map.get_ptr(&3) {
            map.move_to_head(ptr);
            map.debug_validate();
        }

        map.swap_remove(&2);
        map.debug_validate();

        map.clear();
        map.debug_validate();
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_with_ptr() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());

        let ptr1 = map.get_ptr(&1).unwrap();
        let mut cursor = map.ptr_cursor_mut(ptr1);

        assert_eq!(cursor.ptr(), ptr1);
        assert_eq!(cursor.current(), Some((&1, &"one".to_string())));

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&2, &"two".to_string())));
        assert_ne!(cursor.ptr(), ptr1);
    }

    #[test]
    #[timeout(1000)]
    fn test_cursor_mut_nonexistent_key() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());

        let mut cursor = map.key_cursor_mut(&999);
        assert_eq!(cursor.current(), None);
        assert_eq!(cursor.ptr(), Ptr::null());

        let old_value = cursor.insert_after_move_to(999, "new".to_string());
        assert_eq!(old_value, None);
        assert_eq!(map.get(&999), Some(&"new".to_string()));
    }

    #[test]
    #[timeout(1000)]
    fn test_comprehensive_ordering_invariants() {
        let mut map = LinkedHashMap::default();

        for i in 1..=5 {
            map.insert_tail(i, i);
        }

        map.insert_head(0, 0);
        map.swap_remove(&3);
        if let Some(ptr) = map.get_ptr(&4) {
            map.move_to_head(ptr);
        }
        map.insert_tail(6, 6);
        if let Some(ptr) = map.get_ptr(&1) {
            map.move_to_tail(ptr);
        }

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        let head_key = map.ptr_get_entry(map.head_ptr()).map(|(k, _)| *k);
        let tail_key = map.ptr_get_entry(map.tail_ptr()).map(|(k, _)| *k);

        assert_eq!(head_key, Some(items[0]));
        assert_eq!(tail_key, Some(items[items.len() - 1]));

        let mut forward_ptrs = Vec::new();
        let mut current_ptr = map.head_ptr();
        while current_ptr != Ptr::null() {
            forward_ptrs.push(current_ptr);
            current_ptr = map.next_ptr(current_ptr).unwrap_or(Ptr::null());
        }

        let mut backward_ptrs = Vec::new();
        let mut current_ptr = map.tail_ptr();
        while current_ptr != Ptr::null() {
            backward_ptrs.push(current_ptr);
            current_ptr = map.prev_ptr(current_ptr).unwrap_or(Ptr::null());
        }

        backward_ptrs.reverse();
        assert_eq!(forward_ptrs, backward_ptrs);
        assert_eq!(forward_ptrs.len(), map.len());
    }
}
