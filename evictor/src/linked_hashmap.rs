mod arena;

use std::{
    hash::Hash,
    ops::{
        Index,
        IndexMut,
    },
};

use hashbrown::{
    HashTable,
    hash_table,
};

pub use crate::linked_hashmap::arena::Ptr;
use crate::{
    RandomState,
    linked_hashmap::arena::{
        Arena,
        LLData,
    },
};

#[derive(Clone)]
pub struct LinkedHashMap<K, T> {
    head: Ptr,
    tail: Ptr,
    nodes: Arena<K, T>,
    table: HashTable<Ptr>,
    hasher: RandomState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct RemovedEntry<K, T> {
    pub key: K,
    pub value: T,
    pub prev: Ptr,
    pub next: Ptr,
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

        for ptr in self.table.iter().copied() {
            let Some(key) = self.key_for_ptr(ptr) else {
                continue;
            };
            let Some(value) = self.ptr_get(ptr) else {
                continue;
            };
            let next = self.next_ptr(ptr).unwrap_or_default();
            let prev = self.prev_ptr(ptr).unwrap_or_default();
            let next_key = self.key_for_ptr(next);
            let prev_key = self.key_for_ptr(prev);

            entries.push(Entry {
                key,
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
            nodes: Arena::new(),
            table: HashTable::new(),
            hasher: RandomState::default(),
        }
    }
}

impl<K, T> LinkedHashMap<K, T> {
    pub fn with_capacity(capacity: usize) -> Self {
        LinkedHashMap {
            head: Ptr::null(),
            tail: Ptr::null(),
            nodes: Arena::with_capacity(capacity),
            table: HashTable::with_capacity(capacity),
            hasher: RandomState::default(),
        }
    }

    pub fn index_ptr_unstable(&self, index: usize) -> Option<Ptr> {
        self.nodes.ptr_for_index(index)
    }

    pub fn ptr_index_unstable(&self, ptr: Ptr) -> Option<usize> {
        if ptr.is_null() {
            return None;
        }
        self.nodes.index_for_ptr(ptr)
    }

    pub fn move_after(&mut self, moved: Ptr, after: Ptr) -> Option<()> {
        if moved == after {
            return None;
        }

        let after = after.optional()?;
        let moved = moved.optional()?;

        if after == self.tail && moved == self.head {
            self.tail = moved;
            self.head = self.nodes.links(moved).next();
            return Some(());
        }

        if self.nodes.links(after).next() == moved {
            return None;
        }

        let needs_next = self.nodes.links(moved).prev();
        let needs_prev = self.nodes.links(moved).next();
        let also_needs_prev = self.nodes.links(after).next();

        *self.nodes.links_mut(moved).next_mut() = also_needs_prev;
        *self.nodes.links_mut(moved).prev_mut() = after;
        *self.nodes.links_mut(after).next_mut() = moved;

        *self.nodes.links_mut(also_needs_prev).prev_mut() = moved;
        *self.nodes.links_mut(needs_next).next_mut() = needs_prev;
        *self.nodes.links_mut(needs_prev).prev_mut() = needs_next;

        if self.head == moved {
            self.head = needs_prev.or(needs_next);
        }
        if self.tail == moved {
            self.tail = needs_next.or(needs_prev);
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

        let before = before.optional()?;
        let moved = moved.optional()?;

        if before == self.head && moved == self.tail {
            self.head = moved;
            self.tail = self.nodes.links(moved).prev();
            return Some(());
        }

        if self.nodes.links(before).prev() == moved {
            return None;
        }

        let needs_next = self.nodes.links(moved).prev();
        let needs_prev = self.nodes.links(moved).next();
        let also_needs_next = self.nodes.links(before).prev();

        *self.nodes.links_mut(moved).next_mut() = before;
        *self.nodes.links_mut(moved).prev_mut() = also_needs_next;
        *self.nodes.links_mut(before).prev_mut() = moved;

        *self.nodes.links_mut(also_needs_next).next_mut() = moved;
        *self.nodes.links_mut(needs_next).next_mut() = needs_prev;
        *self.nodes.links_mut(needs_prev).prev_mut() = needs_next;

        if self.head == moved {
            self.head = needs_prev.or(needs_next);
        }
        if self.tail == moved {
            self.tail = needs_next.or(needs_prev);
        }

        if self.head == before {
            self.head = moved;
        }

        Some(())
    }

    pub fn link_as_head(&mut self, ptr: Ptr) -> Option<()> {
        self.link_node(ptr, self.tail, self.head, true)
    }

    pub fn link_as_tail(&mut self, ptr: Ptr) -> Option<()> {
        self.link_node(ptr, self.tail, self.head, false)
    }

    pub fn link_node(&mut self, ptr: Ptr, prev: Ptr, next: Ptr, as_head: bool) -> Option<()> {
        debug_assert_ne!(ptr, Ptr::null(), "Cannot link null pointer");

        if self.head == Ptr::null() && self.tail == Ptr::null() {
            debug_assert_eq!(prev, Ptr::null());
            debug_assert_eq!(next, Ptr::null());

            self.head = ptr;
            self.tail = ptr;
            *self.nodes.links_mut(ptr).next_mut() = ptr;
            *self.nodes.links_mut(ptr).prev_mut() = ptr;
            return Some(());
        }

        let prev = if prev == Ptr::null() {
            self.nodes.links(next).prev()
        } else {
            prev
        };
        let next = if next == Ptr::null() {
            self.nodes.links(prev).next()
        } else {
            next
        };

        *self.nodes.links_mut(prev).next_mut() = ptr;
        *self.nodes.links_mut(next).prev_mut() = ptr;

        *self.nodes.links_mut(ptr).prev_mut() = prev;
        *self.nodes.links_mut(ptr).next_mut() = next;

        if as_head && self.head == next {
            self.head = ptr;
        } else if !as_head && self.tail == prev {
            self.tail = ptr;
        }

        Some(())
    }

    pub fn move_to_tail(&mut self, moved: Ptr) -> Option<()> {
        self.move_after(moved, self.tail_ptr())
    }

    pub fn move_to_head(&mut self, moved: Ptr) -> Option<()> {
        self.move_before(moved, self.head_ptr())
    }

    pub fn ptr_cursor_mut(&'_ mut self, ptr: Ptr) -> CursorMut<'_, K, T> {
        CursorMut { ptr, map: self }
    }

    pub fn next_ptr(&self, ptr: Ptr) -> Option<Ptr> {
        Some(self.nodes.links(ptr.optional()?).next())
    }

    pub fn prev_ptr(&self, ptr: Ptr) -> Option<Ptr> {
        Some(self.nodes.links(ptr.optional()?).prev())
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

    #[track_caller]
    pub fn remove_ptr(&mut self, ptr: Ptr) -> Option<RemovedEntry<K, T>> {
        if ptr.is_null() || self.is_empty() {
            return None;
        }

        let slot = self.nodes.free(ptr);
        let prev = slot.prev;
        let next = slot.next;
        let data = slot.into_data();

        match self.table.find_entry(data.hash, move |k| *k == ptr) {
            Ok(occupied) => {
                occupied.remove();
            }
            Err(_) => {
                unreachable!("Pointer not found in table");
            }
        };

        self.finish_removal(ptr, data.key, data.value, prev, next)
    }

    fn finish_removal(
        &mut self,
        ptr: Ptr,
        key: K,
        value: T,
        prev: Ptr,
        next: Ptr,
    ) -> Option<RemovedEntry<K, T>> {
        if self.head == ptr && self.tail == ptr {
            self.head = Ptr::null();
            self.tail = Ptr::null();
            Some(RemovedEntry {
                key,
                value,
                prev: Ptr::null(),
                next: Ptr::null(),
            })
        } else {
            *self.nodes.links_mut(prev).next_mut() = next;
            *self.nodes.links_mut(next).prev_mut() = prev;

            if self.tail == ptr {
                self.tail = prev;
            }
            if self.head == ptr {
                self.head = next;
            }

            Some(RemovedEntry {
                key,
                value,
                prev,
                next,
            })
        }
    }

    pub fn ptr_get(&self, ptr: Ptr) -> Option<&T> {
        Some(&self.nodes[ptr.optional()?].value)
    }

    pub fn ptr_get_entry(&self, ptr: Ptr) -> Option<(&K, &T)> {
        let node = &self.nodes[ptr.optional()?];
        Some((&node.key, &node.value))
    }

    pub fn ptr_get_entry_mut(&mut self, ptr: Ptr) -> Option<(&K, &mut T)> {
        let node = &mut self.nodes[ptr.optional()?];
        Some((&node.key, &mut node.value))
    }

    pub fn ptr_get_mut(&mut self, ptr: Ptr) -> Option<&mut T> {
        Some(&mut self.nodes[ptr.optional()?].value)
    }

    pub fn key_for_ptr(&self, ptr: Ptr) -> Option<&K> {
        Some(&self.nodes[ptr.optional()?].key)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.table.clear();
        self.nodes.clear();
        self.head = Ptr::null();
        self.tail = Ptr::null();
    }

    pub fn shrink_to_fit(&mut self) {
        self.table.shrink_to_fit(|k| self.nodes[*k].hash);
        self.nodes.shrink_to_fit();
    }

    pub fn iter(&'_ self) -> Iter<'_, K, T> {
        Iter {
            ptr: self.head,
            end: self.head,
            map: self,
        }
    }

    pub fn iter_rev(&'_ self) -> IterRev<'_, K, T> {
        IterRev {
            ptr: self.tail,
            end: self.tail,
            map: self,
        }
    }

    pub fn into_iter(self) -> IntoIter<K, T> {
        IntoIter {
            nodes: self.nodes,
            end: self.head,
            ptr: self.head,
        }
    }

    pub fn into_iter_rev(self) -> IntoIterRev<K, T> {
        IntoIterRev {
            nodes: self.nodes,
            end: self.tail,
            ptr: self.tail,
        }
    }

    pub fn contains_ptr(&self, ptr: Ptr) -> bool {
        self.nodes.is_occupied(ptr)
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
            Some(self.tail),
            "Head should link to tail"
        );
        assert_eq!(
            self.next_ptr(self.tail),
            Some(self.head),
            "Tail should link to head"
        );
        assert_eq!(
            self.len(),
            self.table.len(),
            "Links and map should have the same length"
        );

        for ptr in self.table.iter().copied() {
            let node = &self.nodes.links(ptr);

            if ptr == self.head {
                assert_eq!(
                    node.prev(),
                    self.tail,
                    "Head pointer should not have a previous link"
                );
            } else {
                assert_ne!(
                    node.prev(),
                    Ptr::null(),
                    "Non-head pointer should have a previous link: {ptr:?} head: {:?}",
                    self.head
                );
            }

            if ptr == self.tail {
                assert_eq!(
                    node.next(),
                    self.head,
                    "Tail pointer should not have a next link"
                );
            } else {
                assert_ne!(
                    node.next(),
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
        let hash = self.hasher.hash_one(key);
        let ptr = self
            .table
            .find(hash, |k| self.nodes[*k].key == *key)
            .copied()
            .unwrap_or_default();
        CursorMut { ptr, map: self }
    }

    pub fn entry(&'_ mut self, key: K) -> Entry<'_, K, T> {
        let hash = self.hasher.hash_one(&key);
        match self.table.entry(
            hash,
            |k| self.nodes[*k].key == key,
            |idx| self.nodes[*idx].hash,
        ) {
            hash_table::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry {
                node: &mut self.nodes[*entry.get()],
                entry,
            }),
            hash_table::Entry::Vacant(entry) => Entry::Vacant(VacantEntry {
                entry,
                key,
                hash,
                nodes: &mut self.nodes,
                head: &mut self.head,
                tail: &mut self.tail,
            }),
        }
    }

    #[track_caller]
    pub fn remove(&mut self, key: &K) -> Option<(Ptr, RemovedEntry<K, T>)> {
        if self.is_empty() {
            return None;
        }

        let hash = self.hasher.hash_one(key);
        match self.table.find_entry(hash, |k| self.nodes[*k].key == *key) {
            Ok(occupied) => {
                let ptr = *occupied.get();
                let slot = self.nodes.free(ptr);
                let prev = slot.prev;
                let next = slot.next;
                let LLData { key, value, .. } = slot.into_data();

                occupied.remove();
                self.finish_removal(ptr, key, value, prev, next)
                    .map(|entry| (ptr, entry))
            }
            Err(_) => None,
        }
    }

    pub fn get_ptr(&self, key: &K) -> Option<Ptr> {
        let hash = self.hasher.hash_one(key);
        self.table
            .find(hash, |k| self.nodes[*k].key == *key)
            .copied()
    }

    pub fn get(&self, key: &K) -> Option<&T> {
        let ptr = self.get_ptr(key)?;
        self.ptr_get(ptr)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut T> {
        let ptr = self.get_ptr(key)?;
        self.ptr_get_mut(ptr)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.get_ptr(key).is_some()
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
        let ptr = self.ptr.or(self.map.tail);
        match self.map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let map_ptr = occupied_entry.ptr();
                self.map.move_after(map_ptr, ptr);
                self.ptr = map_ptr;
                Some(std::mem::replace(self.map.ptr_get_mut(map_ptr)?, value))
            }
            Entry::Vacant(vacant_entry) => {
                self.ptr = vacant_entry.insert_after(value, ptr);
                None
            }
        }
    }

    pub fn insert_before_move_to(&mut self, key: K, value: T) -> Option<T> {
        let ptr = self.ptr.or(self.map.head);
        match self.map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let map_ptr = occupied_entry.ptr();
                self.map.move_before(map_ptr, ptr);
                self.ptr = map_ptr;
                Some(std::mem::replace(self.map.ptr_get_mut(map_ptr)?, value))
            }
            Entry::Vacant(vacant_entry) => {
                self.ptr = vacant_entry.insert_before(value, ptr);
                None
            }
        }
    }

    pub fn iter(&self) -> Iter<'_, K, T> {
        Iter {
            ptr: self.ptr,
            end: self.map.head,
            map: self.map,
        }
    }

    pub fn iter_rev(&self) -> IterRev<'_, K, T> {
        IterRev {
            ptr: self.ptr,
            end: self.map.tail,
            map: self.map,
        }
    }

    pub fn get_ptr(&self, key: &K) -> Option<Ptr> {
        self.map.get_ptr(key)
    }

    pub fn remove_prev(&mut self) -> Option<RemovedEntry<K, T>> {
        self.map.remove_ptr(self.map.prev_ptr(self.ptr)?)
    }

    pub fn remove_next(&mut self) -> Option<RemovedEntry<K, T>> {
        self.map.remove_ptr(self.map.next_ptr(self.ptr)?)
    }

    pub fn remove(self) -> Option<RemovedEntry<K, T>> {
        self.map.remove_ptr(self.ptr)
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

    pub fn at_tail(&self) -> bool {
        self.ptr == self.map.tail
    }

    pub fn at_head(&self) -> bool {
        self.ptr == self.map.head
    }

    pub fn current(&self) -> Option<(&K, &T)> {
        self.map.ptr_get_entry(self.ptr)
    }

    pub fn current_mut(&mut self) -> Option<(&K, &mut T)> {
        self.map.ptr_get_entry_mut(self.ptr)
    }

    pub fn next_ptr(&self) -> Option<Ptr> {
        self.map.next_ptr(self.ptr)
    }

    pub fn next(&self) -> Option<(&K, &T)> {
        let ptr = self.next_ptr()?;
        self.map.ptr_get_entry(ptr)
    }

    pub fn next_mut(&mut self) -> Option<(&K, &mut T)> {
        let ptr = self.next_ptr()?;
        self.map.ptr_get_entry_mut(ptr)
    }

    pub fn prev_ptr(&self) -> Option<Ptr> {
        self.map.prev_ptr(self.ptr)
    }

    pub fn prev(&self) -> Option<(&K, &T)> {
        let ptr = self.prev_ptr()?;
        self.map.ptr_get_entry(ptr)
    }

    pub fn prev_mut(&mut self) -> Option<(&K, &mut T)> {
        let ptr = self.prev_ptr()?;
        self.map.ptr_get_entry_mut(ptr)
    }
}

impl<K, T> Index<Ptr> for LinkedHashMap<K, T> {
    type Output = T;

    fn index(&self, index: Ptr) -> &Self::Output {
        &self.nodes[index].value
    }
}

impl<K, T> IndexMut<Ptr> for LinkedHashMap<K, T> {
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        &mut self.nodes[index].value
    }
}

#[doc(hidden)]
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

#[doc(hidden)]
pub struct OccupiedEntry<'a, K, V> {
    entry: hash_table::OccupiedEntry<'a, Ptr>,
    node: &'a mut LLData<K, V>,
}

impl<K, V> OccupiedEntry<'_, K, V> {
    pub fn get(&self) -> &V {
        &self.node.value
    }

    pub fn get_mut(&mut self) -> &mut V {
        &mut self.node.value
    }

    pub fn insert_no_move(self, value: V) -> V {
        std::mem::replace(&mut self.node.value, value)
    }

    pub fn ptr(&self) -> Ptr {
        *self.entry.get()
    }

    pub fn key(&self) -> &K {
        &self.node.key
    }
}

#[doc(hidden)]
pub struct VacantEntry<'a, K, V> {
    key: K,
    hash: u64,
    entry: hash_table::VacantEntry<'a, Ptr>,
    nodes: &'a mut Arena<K, V>,
    head: &'a mut Ptr,
    tail: &'a mut Ptr,
}

impl<K: Hash + Eq, V> VacantEntry<'_, K, V> {
    pub fn insert_tail(self, value: V) -> Ptr {
        let after = *self.tail;
        self.insert_after(value, after)
    }

    pub fn push_unlinked(self, value: V) -> Ptr {
        let ptr = self.nodes.alloc(self.key, value, self.hash);
        self.entry.insert(ptr);
        ptr
    }

    pub fn insert_after(self, value: V, after: Ptr) -> Ptr {
        if !self.head.is_null() && !self.tail.is_null() {
            let after_next = self.nodes.links(after).next();
            let ptr = self.nodes.alloc(self.key, value, self.hash);
            *self.nodes.links_mut(ptr).prev_mut() = after;
            *self.nodes.links_mut(ptr).next_mut() = after_next;

            *self.nodes.links_mut(after).next_mut() = ptr;
            *self.nodes.links_mut(after_next).prev_mut() = ptr;
            self.entry.insert(ptr);

            if *self.tail == after {
                *self.tail = ptr;
            }
            ptr
        } else {
            let ptr = self.nodes.alloc(self.key, value, self.hash);
            *self.nodes.links_mut(ptr).prev_mut() = ptr;
            *self.nodes.links_mut(ptr).next_mut() = ptr;

            *self.head = ptr;
            *self.tail = *self.head;
            self.entry.insert(ptr);
            *self.head
        }
    }

    pub fn insert_head(self, value: V) -> Ptr {
        let ptr = *self.head;
        self.insert_before(value, ptr)
    }

    pub fn insert_before(self, value: V, before: Ptr) -> Ptr {
        if !self.head.is_null() && !self.tail.is_null() {
            let before_prev = self.nodes.links(before).prev();
            let ptr = self.nodes.alloc(self.key, value, self.hash);
            *self.nodes.links_mut(ptr).next_mut() = before;
            *self.nodes.links_mut(ptr).prev_mut() = before_prev;

            *self.nodes.links_mut(before).prev_mut() = ptr;
            *self.nodes.links_mut(before_prev).next_mut() = ptr;
            self.entry.insert(ptr);

            if *self.head == before {
                *self.head = ptr;
            }
            ptr
        } else {
            let ptr = self.nodes.alloc(self.key, value, self.hash);
            *self.nodes.links_mut(ptr).prev_mut() = ptr;
            *self.nodes.links_mut(ptr).next_mut() = ptr;

            *self.head = ptr;
            *self.tail = *self.head;
            self.entry.insert(ptr);
            *self.head
        }
    }

    pub fn into_key(self) -> K {
        self.key
    }

    pub fn key(&self) -> &K {
        &self.key
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct Iter<'a, K, T> {
    ptr: Ptr,
    end: Ptr,
    map: &'a LinkedHashMap<K, T>,
}

impl<'a, K, T> Iterator for Iter<'a, K, T> {
    type Item = (&'a K, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.ptr;
        self.ptr = self.map.next_ptr(ptr).unwrap_or_default();
        if self.ptr == self.end {
            self.ptr = Ptr::null();
        }
        self.map.ptr_get_entry(ptr)
    }
}

#[doc(hidden)]
pub struct IterRev<'a, K, T> {
    ptr: Ptr,
    end: Ptr,
    map: &'a LinkedHashMap<K, T>,
}

impl<'a, K, T> Iterator for IterRev<'a, K, T> {
    type Item = (&'a K, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.ptr;
        self.ptr = self.map.prev_ptr(ptr).unwrap_or_default();
        if self.ptr == self.end {
            self.ptr = Ptr::null();
        }
        self.map.ptr_get_entry(ptr)
    }
}

#[doc(hidden)]
pub struct IntoIter<K, T> {
    nodes: Arena<K, T>,
    end: Ptr,
    ptr: Ptr,
}

impl<K, T> Iterator for IntoIter<K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            return None;
        }

        let ptr = self.ptr;
        let slot = self.nodes.free(ptr);
        let next = slot.next;
        let LLData { key, value, .. } = slot.into_data();
        self.ptr = if next == self.end { Ptr::null() } else { next };
        Some((key, value))
    }
}

#[doc(hidden)]
pub struct IntoIterRev<K, T> {
    ptr: Ptr,
    end: Ptr,
    nodes: Arena<K, T>,
}

impl<K, T> Iterator for IntoIterRev<K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            return None;
        }

        let ptr = self.ptr;
        let slot = self.nodes.free(ptr);
        let prev = slot.prev;
        let LLData { key, value, .. } = slot.into_data();

        self.ptr = if prev == self.end { Ptr::null() } else { prev };
        Some((key, value))
    }
}

#[cfg(test)]
mod tests {
    use ntest::timeout;

    use super::*;

    #[test]
    #[timeout(5000)]
    fn test_new_and_default() {
        let map: LinkedHashMap<i32, String> = LinkedHashMap::default();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.head_ptr(), Ptr::null());
        assert_eq!(map.tail_ptr(), Ptr::null());
    }

    #[test]
    #[timeout(5000)]
    fn test_with_capacity() {
        let map: LinkedHashMap<i32, String> = LinkedHashMap::with_capacity(10);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_remove_by_key() {
        let mut map: LinkedHashMap<i32, String> = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());
        map.insert_tail(3, "three".to_string());

        let removed = map.remove(&2).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 2,
                value: "two".to_string(),
                prev: map.get_ptr(&1).unwrap(),
                next: map.get_ptr(&3).unwrap(),
            }
        );
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key(&2));

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![(&1, &"one".to_string()), (&3, &"three".to_string())]
        );

        let removed = map.remove(&1).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 1,
                value: "one".to_string(),
                prev: map.get_ptr(&3).unwrap(),
                next: map.get_ptr(&3).unwrap(),
            }
        );
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let removed = map.remove(&3).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 3,
                value: "three".to_string(),
                prev: Ptr::null(),
                next: Ptr::null(),
            }
        );
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.head_ptr(), Ptr::null());
        assert_eq!(map.tail_ptr(), Ptr::null());

        let removed = map.remove(&1);
        assert_eq!(removed, None);
    }

    #[test]
    #[timeout(5000)]
    fn test_remove_by_ptr() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());
        map.insert_tail(2, "two".to_string());
        map.insert_tail(3, "three".to_string());

        let removed = map.remove_ptr(map.get_ptr(&2).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 2,
                value: "two".to_string(),
                prev: map.get_ptr(&1).unwrap(),
                next: map.get_ptr(&3).unwrap(),
            })
        );
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key(&2));

        let removed = map.remove_ptr(map.get_ptr(&1).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 1,
                value: "one".to_string(),
                prev: map.get_ptr(&3).unwrap(),
                next: map.get_ptr(&3).unwrap(),
            })
        );
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.get_ptr(&3).unwrap());
        assert_eq!(map.tail_ptr(), map.get_ptr(&3).unwrap());

        let removed = map.remove_ptr(map.get_ptr(&1).unwrap_or(Ptr::null()));
        assert_eq!(removed, None);

        let removed = map.remove_ptr(map.get_ptr(&3).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 3,
                value: "three".to_string(),
                prev: Ptr::null(),
                next: Ptr::null(),
            })
        );
        assert!(map.is_empty());
    }

    #[test]
    #[timeout(5000)]
    fn test_remove_single_element() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(42, "answer".to_string());

        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let removed = map.remove(&42).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 42,
                value: "answer".to_string(),
                prev: Ptr::null(),
                next: Ptr::null(),
            }
        );
        assert!(map.is_empty());
        assert_eq!(map.head_ptr(), Ptr::null());
        assert_eq!(map.tail_ptr(), Ptr::null());
    }

    #[test]
    #[timeout(5000)]
    fn test_remove_head_and_tail() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        let removed = map.remove(&1).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 1,
                value: "value1".to_string(),
                prev: map.get_ptr(&5).unwrap(),
                next: map.get_ptr(&2).unwrap(),
            }
        );
        assert_eq!(map.tail_ptr(), map.get_ptr(&5).unwrap());

        let removed = map.remove(&5).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 5,
                value: "value5".to_string(),
                prev: map.get_ptr(&4).unwrap(),
                next: map.get_ptr(&2).unwrap(),
            }
        );
        assert_eq!(map.len(), 3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 3, 4]);
    }

    #[test]
    #[timeout(5000)]
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
        map.move_to_head(ptr4).unwrap();

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![4, 3, 1, 2]);
        assert_eq!(map.head_ptr(), ptr4);
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
        assert_eq!(map.next_ptr(ptr3), Some(ptr1));
        assert_eq!(map.next_ptr(Ptr::null()), None);

        assert_eq!(map.prev_ptr(ptr1), Some(ptr3));
        assert_eq!(map.prev_ptr(ptr2), Some(ptr1));
        assert_eq!(map.prev_ptr(ptr3), Some(ptr2));
        assert_eq!(map.prev_ptr(Ptr::null()), None);
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_iteration_after_modifications() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        map.remove(&3);

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
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_single_element_iteration() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(42, "answer".to_string());

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&42, &"answer".to_string())]);

        let items: Vec<_> = map.iter_rev().collect();
        assert_eq!(items, vec![(&42, &"answer".to_string())]);
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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
            })
        );
        assert!(!map.contains_key(&3));

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5]);
    }

    #[test]
    #[timeout(5000)]
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
                prev: map.get_ptr(&2).unwrap(),
                next: map.get_ptr(&2).unwrap(),
            })
        );

        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&1));
        assert_eq!(map.get(&2), Some(&"two".to_string()));
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_edge_cases_invalid_pointers() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one".to_string());

        let invalid_ptr = Ptr::null();

        assert_eq!(map.ptr_get(invalid_ptr), None);
        assert_eq!(map.ptr_get_entry(invalid_ptr), None);
        assert_eq!(map.ptr_get_entry_mut(invalid_ptr), None);
        assert_eq!(map.ptr_get_mut(invalid_ptr), None);
        assert_eq!(map.key_for_ptr(invalid_ptr), None);
        assert_eq!(map.remove_ptr(invalid_ptr), None);
        assert!(!map.contains_ptr(invalid_ptr));

        map.move_to_head(invalid_ptr);
        map.move_to_tail(invalid_ptr);
        map.move_after(invalid_ptr, invalid_ptr);
        map.move_before(invalid_ptr, invalid_ptr);

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&"one".to_string()));
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_iteration_consistency_after_modifications() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, i * 10);
        }

        map.remove(&3);
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
    #[timeout(5000)]
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
    #[timeout(5000)]
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

        map.remove(&2);
        map.debug_validate();

        map.clear();
        map.debug_validate();
    }

    #[test]
    #[timeout(5000)]
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
    #[timeout(5000)]
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
    #[timeout(5000)]
    fn test_comprehensive_ordering_invariants() {
        let mut map = LinkedHashMap::default();

        for i in 1..=5 {
            map.insert_tail(i, i);
        }

        map.insert_head(0, 0);
        map.remove(&3);
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
        let mut looped = false;
        while !looped {
            forward_ptrs.push(current_ptr);
            current_ptr = map.next_ptr(current_ptr).unwrap_or(Ptr::null());
            looped = current_ptr == map.head_ptr();
        }

        let mut backward_ptrs = Vec::new();
        let mut current_ptr = map.tail_ptr();
        let mut looped = false;
        while !looped {
            backward_ptrs.push(current_ptr);
            current_ptr = map.prev_ptr(current_ptr).unwrap_or(Ptr::null());
            looped = current_ptr == map.tail_ptr();
        }

        backward_ptrs.reverse();
        assert_eq!(forward_ptrs, backward_ptrs);
        assert_eq!(forward_ptrs.len(), map.len());
    }
}
