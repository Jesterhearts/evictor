use std::{
    num::NonZeroU32,
    ops::{
        Index,
        IndexMut,
    },
};

#[derive(Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
#[repr(transparent)]
pub struct Ptr(NonZeroU32);

impl std::fmt::Debug for Ptr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Ptr::null() {
            write!(f, "Ptr(null)")
        } else {
            write!(f, "Ptr({})", self.0.get() - 1)
        }
    }
}

impl Default for Ptr {
    fn default() -> Self {
        Ptr::null()
    }
}

impl Ptr {
    pub(crate) fn null() -> Self {
        Ptr(NonZeroU32::new(u32::MAX).unwrap())
    }

    pub(crate) fn is_null(&self) -> bool {
        *self == Ptr::null()
    }

    pub(crate) fn unchecked_from(index: usize) -> Self {
        debug_assert!(
            index < u32::MAX as usize - 1,
            "Index too large to fit in Ptr: {index}"
        );
        Ptr(NonZeroU32::new((index as u32).wrapping_add(1)).unwrap())
    }

    pub(crate) fn unchecked_get(self) -> usize {
        self.0.get() as usize - 1
    }

    pub(crate) fn get(self) -> Option<usize> {
        if self.is_null() {
            None
        } else {
            Some(self.0.get() as usize - 1)
        }
    }

    pub(crate) fn optional(self) -> Option<Ptr> {
        if self.is_null() { None } else { Some(self) }
    }

    pub(crate) fn or(&self, tail_ptr: Ptr) -> Ptr {
        if self.is_null() { tail_ptr } else { *self }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LLData<K, T> {
    back_ptr_index: Ptr,
    pub(crate) hash: u64,
    pub(crate) key: K,
    pub(crate) value: T,
}

#[derive(Debug, Clone, Copy)]
enum DataOrFree<K, T> {
    Free,
    Data(LLData<K, T>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LLSlot<K, T> {
    pub(crate) prev: Ptr,
    pub(crate) next: Ptr,
    data: DataOrFree<K, T>,
}

impl<K, T> LLSlot<K, T> {
    pub(crate) fn prev(&self) -> Ptr {
        self.prev
    }

    pub(crate) fn prev_mut(&mut self) -> &mut Ptr {
        &mut self.prev
    }

    pub(crate) fn next(&self) -> Ptr {
        self.next
    }

    pub(crate) fn next_mut(&mut self) -> &mut Ptr {
        &mut self.next
    }

    pub(crate) fn into_data(self) -> LLData<K, T> {
        match self.data {
            DataOrFree::Data(data) => data,
            DataOrFree::Free => unreachable!("Attempted to extract data from a free slot"),
        }
    }

    pub(crate) fn data(&self) -> &LLData<K, T> {
        match &self.data {
            DataOrFree::Data(data) => data,
            DataOrFree::Free => unreachable!("Attempted to access data of uninitialized slot"),
        }
    }

    pub(crate) fn data_mut(&mut self) -> &mut LLData<K, T> {
        match &mut self.data {
            DataOrFree::Data(data) => data,
            DataOrFree::Free => unreachable!("Attempted to access data of uninitialized slot"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Arena<K, T> {
    nodes: Vec<LLSlot<K, T>>,
    back_ptrs: Vec<Ptr>,
    free_head: Ptr,
}

impl<K, T> Arena<K, T> {
    pub(crate) fn new() -> Self {
        Arena {
            nodes: Vec::new(),
            back_ptrs: Vec::new(),
            free_head: Ptr::null(),
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        assert!(capacity < u32::MAX as usize - 1, "Capacity too large");
        let mut nodes = Vec::with_capacity(capacity);
        for i in 0..capacity {
            nodes.push(LLSlot {
                prev: Ptr::null(),
                next: if i + 1 < capacity {
                    Ptr::unchecked_from(i + 1)
                } else {
                    Ptr::null()
                },
                data: DataOrFree::Free,
            });
        }
        Arena {
            nodes,
            back_ptrs: Vec::with_capacity(capacity),
            free_head: Ptr::unchecked_from(0),
        }
    }

    pub(crate) fn links(&self, ptr: Ptr) -> &LLSlot<K, T> {
        &self.nodes[ptr.unchecked_get()]
    }

    pub(crate) fn links_mut(&mut self, ptr: Ptr) -> &mut LLSlot<K, T> {
        &mut self.nodes[ptr.unchecked_get()]
    }

    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.back_ptrs.clear();
        self.free_head = Ptr::null();
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        // Note: This may not even shrink anything if the arena has free slots. In
        // general, it's not possible to move around the nodes, since there may be
        // external Ptrs pointing to them. So this is the best we can do.
        // It *might* be possible to compact the arena by moving occupied nodes to
        // fill in free slots, but would require keeping a mapping of all moved Ptrs so
        // they can be remapped when calling free/index/etc. That might even *increase*
        // memory used depending on the exact usage pattern and would both add
        // complexity and likely be slower in the happy path.
        self.nodes.shrink_to_fit();
        self.back_ptrs.shrink_to_fit();
    }

    #[inline]
    pub(crate) fn alloc(&mut self, key: K, value: T, hash: u64) -> Ptr {
        if !self.free_head.is_null() {
            let ptr = self.free_head;
            self.free_head = self.nodes[ptr.unchecked_get()].next();
            self.nodes[ptr.unchecked_get()].data = DataOrFree::Data(LLData {
                back_ptr_index: Ptr::unchecked_from(self.back_ptrs.len()),
                key,
                value,
                hash,
            });
            self.back_ptrs.push(ptr);
            ptr
        } else {
            let ptr = Ptr::unchecked_from(self.nodes.len());
            self.nodes.push(LLSlot {
                prev: Ptr::null(),
                next: Ptr::null(),
                data: DataOrFree::Data(LLData {
                    back_ptr_index: Ptr::unchecked_from(self.back_ptrs.len()),
                    key,
                    value,
                    hash,
                }),
            });
            self.back_ptrs.push(ptr);
            ptr
        }
    }

    pub(crate) fn ptr_for_index(&self, index: usize) -> Option<Ptr> {
        self.back_ptrs.get(index).copied()
    }

    pub(crate) fn index_for_ptr(&self, ptr: Ptr) -> Option<usize> {
        if !self.is_occupied(ptr) {
            return None;
        }
        self.nodes[ptr.unchecked_get()].data().back_ptr_index.get()
    }

    pub(crate) fn is_occupied(&self, ptr: Ptr) -> bool {
        if ptr.is_null() {
            return false;
        }
        matches!(self.nodes[ptr.unchecked_get()].data, DataOrFree::Data(_))
    }

    #[inline]
    pub(crate) fn free(&mut self, ptr: Ptr) -> LLSlot<K, T> {
        debug_assert!(
            !ptr.is_null() && ptr.unchecked_get() < self.nodes.len(),
            "Pointer to free must be valid: {ptr:?}"
        );
        let result = std::mem::replace(
            &mut self.nodes[ptr.unchecked_get()],
            LLSlot {
                prev: Ptr::null(),
                next: self.free_head,
                data: DataOrFree::Free,
            },
        );
        self.free_head = ptr;

        let back_ptr_index = result.data().back_ptr_index;
        self.back_ptrs.swap_remove(back_ptr_index.unchecked_get());
        if back_ptr_index.unchecked_get() < self.back_ptrs.len() {
            let swapped_ptr = self.back_ptrs[back_ptr_index.unchecked_get()];
            self.nodes[swapped_ptr.unchecked_get()]
                .data_mut()
                .back_ptr_index = back_ptr_index;
        }

        result
    }
}

impl<K, T> Index<Ptr> for Arena<K, T> {
    type Output = LLData<K, T>;

    fn index(&self, index: Ptr) -> &Self::Output {
        self.nodes[index.unchecked_get()].data()
    }
}

impl<K, T> IndexMut<Ptr> for Arena<K, T> {
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        self.nodes[index.unchecked_get()].data_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptr_null() {
        let null_ptr = Ptr::null();
        assert!(null_ptr.is_null());
        assert_eq!(null_ptr.get(), None);
        assert_eq!(null_ptr.optional(), None);
    }

    #[test]
    fn test_ptr_non_null() {
        let ptr = Ptr::unchecked_from(42);
        assert!(!ptr.is_null());
        assert_eq!(ptr.get(), Some(42));
        assert_eq!(ptr.optional(), Some(ptr));
        assert_eq!(ptr.unchecked_get(), 42);
    }

    #[test]
    fn test_ptr_or() {
        let null_ptr = Ptr::null();
        let some_ptr = Ptr::unchecked_from(10);
        let other_ptr = Ptr::unchecked_from(20);

        assert_eq!(null_ptr.or(some_ptr), some_ptr);
        assert_eq!(some_ptr.or(other_ptr), some_ptr);
    }

    #[test]
    fn test_ptr_debug() {
        let null_ptr = Ptr::null();
        let some_ptr = Ptr::unchecked_from(42);

        assert_eq!(format!("{:?}", null_ptr), "Ptr(null)");
        assert_eq!(format!("{:?}", some_ptr), "Ptr(42)");
    }

    #[test]
    fn test_ptr_default() {
        let default_ptr: Ptr = Default::default();
        assert!(default_ptr.is_null());
    }

    #[test]
    fn test_ptr_equality() {
        let ptr1 = Ptr::unchecked_from(42);
        let ptr2 = Ptr::unchecked_from(42);
        let ptr3 = Ptr::unchecked_from(43);

        assert_eq!(ptr1, ptr2);
        assert_ne!(ptr1, ptr3);
    }

    #[test]
    fn test_arena_new() {
        let arena: Arena<i32, String> = Arena::new();
        assert_eq!(arena.nodes.len(), 0);
        assert_eq!(arena.back_ptrs.len(), 0);
        assert!(arena.free_head.is_null());
    }

    #[test]
    fn test_arena_with_capacity() {
        let arena: Arena<i32, String> = Arena::with_capacity(10);
        assert_eq!(arena.nodes.capacity(), 10);
        assert_eq!(arena.back_ptrs.capacity(), 10);
    }

    #[test]
    #[should_panic(expected = "Capacity too large")]
    fn test_arena_with_capacity_too_large() {
        Arena::<i32, String>::with_capacity(usize::MAX);
    }

    #[test]
    fn test_arena_alloc_single() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(42, "hello".to_string(), 12345);

        assert!(!ptr.is_null());
        assert!(arena.is_occupied(ptr));
        assert_eq!(arena.nodes.len(), 1);
        assert_eq!(arena.back_ptrs.len(), 1);

        let data = &arena[ptr];
        assert_eq!(data.key, 42);
        assert_eq!(data.value, "hello");
        assert_eq!(data.hash, 12345);
    }

    #[test]
    fn test_arena_alloc_multiple() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111);
        let ptr2 = arena.alloc(2, "two".to_string(), 222);
        let ptr3 = arena.alloc(3, "three".to_string(), 333);

        assert_ne!(ptr1, ptr2);
        assert_ne!(ptr2, ptr3);
        assert_ne!(ptr1, ptr3);

        assert!(arena.is_occupied(ptr1));
        assert!(arena.is_occupied(ptr2));
        assert!(arena.is_occupied(ptr3));

        assert_eq!(arena[ptr1].key, 1);
        assert_eq!(arena[ptr2].key, 2);
        assert_eq!(arena[ptr3].key, 3);
    }

    #[test]
    fn test_arena_free_and_reuse() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111);
        let ptr2 = arena.alloc(2, "two".to_string(), 222);

        assert!(arena.is_occupied(ptr1));
        assert!(arena.is_occupied(ptr2));

        let data = arena.free(ptr1);
        assert_eq!(data.data().key, 1);
        assert_eq!(data.data().value, "one");
        assert!(!arena.is_occupied(ptr1));
        assert!(arena.is_occupied(ptr2));

        let ptr3 = arena.alloc(3, "three".to_string(), 333);
        assert_eq!(ptr3, ptr1);
        assert!(arena.is_occupied(ptr3));
        assert_eq!(arena[ptr3].key, 3);
    }

    #[test]
    fn test_arena_index_operations() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(42, "hello".to_string(), 12345);

        let data = &arena[ptr];
        assert_eq!(data.key, 42);
        assert_eq!(data.value, "hello");

        arena[ptr].value = "world".to_string();
        assert_eq!(arena[ptr].value, "world");
    }

    #[test]
    fn test_arena_ptr_for_index() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111);
        let ptr2 = arena.alloc(2, "two".to_string(), 222);

        assert_eq!(arena.ptr_for_index(0), Some(ptr1));
        assert_eq!(arena.ptr_for_index(1), Some(ptr2));
        assert_eq!(arena.ptr_for_index(2), None);
    }

    #[test]
    fn test_arena_index_for_ptr() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111);
        let ptr2 = arena.alloc(2, "two".to_string(), 222);

        assert_eq!(arena.index_for_ptr(ptr1), Some(0));
        assert_eq!(arena.index_for_ptr(ptr2), Some(1));
        assert_eq!(arena.index_for_ptr(Ptr::null()), None);
    }

    #[test]
    fn test_arena_index_stability_after_free() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111);
        let ptr2 = arena.alloc(2, "two".to_string(), 222);
        let ptr3 = arena.alloc(3, "three".to_string(), 333);

        arena.free(ptr2);

        assert_eq!(arena.index_for_ptr(ptr1), Some(0));
        assert_eq!(arena.index_for_ptr(ptr3), Some(1));
        assert_eq!(arena.index_for_ptr(ptr2), None);

        assert_eq!(arena.ptr_for_index(0), Some(ptr1));
        assert_eq!(arena.ptr_for_index(1), Some(ptr3));
    }

    #[test]
    fn test_arena_links() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(42, "hello".to_string(), 12345);

        let links = arena.links(ptr);
        assert!(links.prev().is_null());
        assert!(links.next().is_null());

        let links_mut = arena.links_mut(ptr);
        *links_mut.prev_mut() = Ptr::unchecked_from(10);
        *links_mut.next_mut() = Ptr::unchecked_from(20);

        let links = arena.links(ptr);
        assert_eq!(links.prev(), Ptr::unchecked_from(10));
        assert_eq!(links.next(), Ptr::unchecked_from(20));
    }

    #[test]
    fn test_arena_clear() {
        let mut arena = Arena::new();
        arena.alloc(1, "one".to_string(), 111);
        arena.alloc(2, "two".to_string(), 222);

        assert_eq!(arena.nodes.len(), 2);
        assert_eq!(arena.back_ptrs.len(), 2);

        arena.clear();

        assert_eq!(arena.nodes.len(), 0);
        assert_eq!(arena.back_ptrs.len(), 0);
        assert!(arena.free_head.is_null());
    }

    #[test]
    fn test_arena_clone() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111);
        let ptr2 = arena.alloc(2, "two".to_string(), 222);

        arena.links_mut(ptr1).next = ptr2;
        arena.links_mut(ptr2).prev = ptr1;

        let cloned_arena = arena.clone();

        assert_eq!(cloned_arena.nodes.len(), arena.nodes.len());
        assert_eq!(cloned_arena.back_ptrs.len(), arena.back_ptrs.len());
        assert_eq!(cloned_arena.free_head, arena.free_head);

        assert_eq!(cloned_arena[ptr1].key, arena[ptr1].key);
        assert_eq!(cloned_arena[ptr1].value, arena[ptr1].value);
        assert_eq!(cloned_arena[ptr2].key, arena[ptr2].key);
        assert_eq!(cloned_arena[ptr2].value, arena[ptr2].value);

        assert_eq!(cloned_arena.links(ptr1).next, ptr2);
        assert_eq!(cloned_arena.links(ptr2).prev, ptr1);
    }

    #[test]
    fn test_arena_clone_with_free_slots() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111);
        let ptr2 = arena.alloc(2, "two".to_string(), 222);
        let ptr3 = arena.alloc(3, "three".to_string(), 333);

        arena.free(ptr2);

        let cloned_arena = arena.clone();

        assert!(cloned_arena.is_occupied(ptr1));
        assert!(!cloned_arena.is_occupied(ptr2));
        assert!(cloned_arena.is_occupied(ptr3));

        assert_eq!(cloned_arena.free_head, arena.free_head);
    }

    #[test]
    #[should_panic]
    fn test_arena_index_unoccupied_ptr() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(1, "one".to_string(), 111);
        arena.free(ptr);
        let _ = &arena[ptr];
    }

    #[test]
    #[should_panic]
    fn test_arena_index_mut_unoccupied_ptr() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(1, "one".to_string(), 111);
        arena.free(ptr);
        let _ = &mut arena[ptr];
    }

    #[test]
    #[should_panic]
    fn test_arena_free_unoccupied_ptr() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(1, "one".to_string(), 111);
        arena.free(ptr);
        arena.free(ptr);
    }

    #[test]
    #[should_panic]
    fn test_arena_free_null_ptr() {
        let mut arena = Arena::<i32, i32>::new();
        arena.free(Ptr::null());
    }

    #[test]
    fn test_arena_is_occupied_null_ptr() {
        let arena: Arena<i32, String> = Arena::new();
        assert!(!arena.is_occupied(Ptr::null()));
    }

    #[test]
    fn test_niche_optimization() {
        use std::mem::size_of;
        assert_eq!(size_of::<Ptr>(), size_of::<u32>());
        assert_eq!(
            size_of::<DataOrFree<String, String>>(),
            size_of::<LLData<String, String>>()
        );
        assert_eq!(
            size_of::<LLSlot<String, String>>(),
            size_of::<(Ptr, Ptr, LLData<String, String>)>()
        );
    }
}
