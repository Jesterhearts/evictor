#![no_main]

use std::num::NonZeroUsize;

use evictor::{
    Lfu,
    Lru,
    Mru,
};
use libfuzzer_sys::fuzz_target;

#[derive(Debug)]
enum CacheOperation {
    Insert(u16, u16),
    Get(u16),
    Peek(u16),
    Remove(u16),
    Pop,
    Clear,
    GetOrInsertWith(u16, u16),
    Retain,
    Iter,
}

impl<'a> arbitrary::Arbitrary<'a> for CacheOperation {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        match u.int_in_range(0..=8)? {
            0 => Ok(CacheOperation::Insert(u.arbitrary()?, u.arbitrary()?)),
            1 => Ok(CacheOperation::Get(u.arbitrary()?)),
            2 => Ok(CacheOperation::Peek(u.arbitrary()?)),
            3 => Ok(CacheOperation::Remove(u.arbitrary()?)),
            4 => Ok(CacheOperation::Pop),
            5 => Ok(CacheOperation::Clear),
            6 => Ok(CacheOperation::GetOrInsertWith(
                u.arbitrary()?,
                u.arbitrary()?,
            )),
            7 => Ok(CacheOperation::Retain),
            8 => Ok(CacheOperation::Iter),
            _ => unreachable!(),
        }
    }
}

fuzz_target!(|data: (u16, Vec<CacheOperation>)| {
    let (capacity_raw, operations) = data;

    let capacity = NonZeroUsize::new((capacity_raw % 4).max(1) as usize).unwrap();
    let mut lfu_cache = Lfu::<u16, u16>::new(capacity);
    let mut lru_cache = Lru::<u16, u16>::new(capacity);
    let mut mru_cache = Mru::<u16, u16>::new(capacity);

    let initial_capacity = lfu_cache.capacity();

    #[allow(clippy::len_zero)]
    for op in operations {
        assert!(lfu_cache.len() <= lfu_cache.capacity());
        assert_eq!(lfu_cache.capacity(), initial_capacity);
        assert_eq!(lfu_cache.is_empty(), lfu_cache.len() == 0);

        assert!(lru_cache.len() <= lru_cache.capacity());
        assert_eq!(lru_cache.capacity(), initial_capacity);
        assert_eq!(lru_cache.is_empty(), lru_cache.len() == 0);

        assert!(mru_cache.len() <= mru_cache.capacity());
        assert_eq!(mru_cache.capacity(), initial_capacity);
        assert_eq!(mru_cache.is_empty(), mru_cache.len() == 0);

        match op {
            CacheOperation::Insert(key, value) => {
                let old_len = lfu_cache.len();
                let contained_before = lfu_cache.contains_key(&key);

                lfu_cache.insert(key, value);

                assert!(lfu_cache.len() <= lfu_cache.capacity());

                if !contained_before && old_len < lfu_cache.capacity() {
                    assert_eq!(lfu_cache.len(), old_len + 1);
                }

                assert!(lfu_cache.contains_key(&key));
                assert_eq!(lfu_cache.peek(&key), Some(&value));

                let old_len = lru_cache.len();
                let contained_before = lru_cache.contains_key(&key);

                lru_cache.insert(key, value);

                assert!(lru_cache.len() <= lru_cache.capacity());

                if !contained_before && old_len < lru_cache.capacity() {
                    assert_eq!(lru_cache.len(), old_len + 1);
                }

                assert!(lru_cache.contains_key(&key));
                assert_eq!(lru_cache.peek(&key), Some(&value));

                let old_len = mru_cache.len();
                let contained_before = mru_cache.contains_key(&key);

                mru_cache.insert(key, value);

                assert!(mru_cache.len() <= lfu_cache.capacity());

                if !contained_before && old_len < mru_cache.capacity() {
                    assert_eq!(mru_cache.len(), old_len + 1);
                }

                assert!(mru_cache.contains_key(&key));
                assert_eq!(mru_cache.peek(&key), Some(&value));
            }

            CacheOperation::Get(key) => {
                let contains_before = lfu_cache.contains_key(&key);
                let result = lfu_cache.get(&key);
                let result_exists = result.is_some();

                assert_eq!(lfu_cache.contains_key(&key), contains_before);
                assert_eq!(result_exists, contains_before);

                let contains_before = lfu_cache.contains_key(&key);
                let result = lfu_cache.get(&key).copied();
                let result_exists = result.is_some();

                assert_eq!(lfu_cache.contains_key(&key), contains_before);
                assert_eq!(result_exists, contains_before);

                let contains_before = lru_cache.contains_key(&key);
                let result = lru_cache.get(&key);
                let result_exists = result.is_some();

                assert_eq!(lru_cache.contains_key(&key), contains_before);
                assert_eq!(result_exists, contains_before);

                let contains_before = lru_cache.contains_key(&key);
                let result = lru_cache.get(&key).copied();
                let result_exists = result.is_some();

                assert_eq!(lru_cache.contains_key(&key), contains_before);
                assert_eq!(result_exists, contains_before);

                let contains_before = mru_cache.contains_key(&key);
                let result = mru_cache.get(&key);
                let result_exists = result.is_some();

                assert_eq!(mru_cache.contains_key(&key), contains_before);
                assert_eq!(result_exists, contains_before);

                let contains_before = mru_cache.contains_key(&key);
                let result = mru_cache.get(&key).copied();
                let result_exists = result.is_some();

                assert_eq!(mru_cache.contains_key(&key), contains_before);
                assert_eq!(result_exists, contains_before);
            }

            CacheOperation::Peek(key) => {
                let contains_before = lfu_cache.contains_key(&key);
                let len_before = lfu_cache.len();
                let tail_before = lfu_cache.tail().map(|t| (*t.0, *t.1));
                let result = lfu_cache.peek(&key);

                assert_eq!(lfu_cache.tail().map(|t| (*t.0, *t.1)), tail_before);
                assert_eq!(lfu_cache.contains_key(&key), contains_before);
                assert_eq!(lfu_cache.len(), len_before);
                assert_eq!(result.is_some(), contains_before);

                let contains_before = lru_cache.contains_key(&key);
                let len_before = lru_cache.len();
                let tail_before = lru_cache.tail().map(|t| (*t.0, *t.1));
                let result = lru_cache.peek(&key);

                assert_eq!(lru_cache.tail().map(|t| (*t.0, *t.1)), tail_before);
                assert_eq!(lru_cache.contains_key(&key), contains_before);
                assert_eq!(lru_cache.len(), len_before);
                assert_eq!(result.is_some(), contains_before);

                let contains_before = mru_cache.contains_key(&key);
                let len_before = mru_cache.len();
                let tail_before = mru_cache.tail().map(|t| (*t.0, *t.1));
                let result = mru_cache.peek(&key);

                assert_eq!(mru_cache.tail().map(|t| (*t.0, *t.1)), tail_before);
                assert_eq!(mru_cache.contains_key(&key), contains_before);
                assert_eq!(mru_cache.len(), len_before);
                assert_eq!(result.is_some(), contains_before);
            }

            CacheOperation::Remove(key) => {
                let contains_before = lfu_cache.contains_key(&key);
                let len_before = lfu_cache.len();
                let result = lfu_cache.remove(&key);

                assert!(!lfu_cache.contains_key(&key));
                assert_eq!(result.is_some(), contains_before);

                if contains_before {
                    assert_eq!(lfu_cache.len(), len_before - 1);
                } else {
                    assert_eq!(lfu_cache.len(), len_before);
                }

                let contains_before = lru_cache.contains_key(&key);
                let len_before = lru_cache.len();
                let result = lru_cache.remove(&key);

                assert!(!lru_cache.contains_key(&key));
                assert_eq!(result.is_some(), contains_before);

                if contains_before {
                    assert_eq!(lru_cache.len(), len_before - 1);
                } else {
                    assert_eq!(lru_cache.len(), len_before);
                }

                let contains_before = mru_cache.contains_key(&key);
                let len_before = mru_cache.len();
                let result = mru_cache.remove(&key);

                assert!(!mru_cache.contains_key(&key));
                assert_eq!(result.is_some(), contains_before);

                if contains_before {
                    assert_eq!(mru_cache.len(), len_before - 1);
                } else {
                    assert_eq!(mru_cache.len(), len_before);
                }
            }

            CacheOperation::Pop => {
                let len_before = lfu_cache.len();
                let was_empty = lfu_cache.is_empty();
                let result = lfu_cache.pop();

                if was_empty {
                    assert_eq!(result, None);
                    assert_eq!(lfu_cache.len(), 0);
                } else {
                    assert!(result.is_some());
                    assert_eq!(lfu_cache.len(), len_before - 1);

                    if let Some((key, _)) = result {
                        assert!(!lfu_cache.contains_key(&key));
                    }
                }

                let len_before = lru_cache.len();
                let was_empty = lru_cache.is_empty();
                let result = lru_cache.pop();

                if was_empty {
                    assert_eq!(result, None);
                    assert_eq!(lru_cache.len(), 0);
                } else {
                    assert!(result.is_some());
                    assert_eq!(lru_cache.len(), len_before - 1);

                    if let Some((key, _)) = result {
                        assert!(!lru_cache.contains_key(&key));
                    }
                }

                let len_before = mru_cache.len();
                let was_empty = mru_cache.is_empty();
                let result = mru_cache.pop();

                if was_empty {
                    assert_eq!(result, None);
                    assert_eq!(mru_cache.len(), 0);
                } else {
                    assert!(result.is_some());
                    assert_eq!(mru_cache.len(), len_before - 1);

                    if let Some((key, _)) = result {
                        assert!(!mru_cache.contains_key(&key));
                    }
                }
            }

            CacheOperation::Clear => {
                lfu_cache.clear();
                assert_eq!(lfu_cache.len(), 0);
                assert!(lfu_cache.is_empty());
                assert_eq!(lfu_cache.tail(), None);
                assert_eq!(lfu_cache.capacity(), initial_capacity);

                lru_cache.clear();
                assert_eq!(lru_cache.len(), 0);
                assert!(lru_cache.is_empty());
                assert_eq!(lru_cache.tail(), None);
                assert_eq!(lru_cache.capacity(), initial_capacity);

                mru_cache.clear();
                assert_eq!(mru_cache.len(), 0);
                assert!(mru_cache.is_empty());
                assert_eq!(mru_cache.tail(), None);
                assert_eq!(mru_cache.capacity(), initial_capacity);
            }

            CacheOperation::GetOrInsertWith(key, value) => {
                let contains_before = lfu_cache.contains_key(&key);
                let len_before = lfu_cache.len();

                let result = lfu_cache.get_or_insert_with(key, |_| value);
                let result_value = *result;

                assert!(lfu_cache.contains_key(&key));

                if contains_before {
                    assert_eq!(lfu_cache.len(), len_before);
                } else {
                    assert_eq!(result_value, value);
                    if len_before < lfu_cache.capacity() {
                        assert_eq!(lfu_cache.len(), len_before + 1);
                    }
                }

                let contains_before = lru_cache.contains_key(&key);
                let len_before = lru_cache.len();

                let result = lru_cache.get_or_insert_with(key, |_| value);
                let result_value = *result;

                assert!(lru_cache.contains_key(&key));

                if contains_before {
                    assert_eq!(lru_cache.len(), len_before);
                } else {
                    assert_eq!(result_value, value);
                    if len_before < lru_cache.capacity() {
                        assert_eq!(lru_cache.len(), len_before + 1);
                    }
                }

                let contains_before = mru_cache.contains_key(&key);
                let len_before = mru_cache.len();

                let result = mru_cache.get_or_insert_with(key, |_| value);
                let result_value = *result;

                assert!(mru_cache.contains_key(&key));

                if contains_before {
                    assert_eq!(mru_cache.len(), len_before);
                } else {
                    assert_eq!(result_value, value);
                    if len_before < mru_cache.capacity() {
                        assert_eq!(mru_cache.len(), len_before + 1);
                    }
                }
            }

            CacheOperation::Retain => {
                let len_before = lfu_cache.len();

                let all_before = lfu_cache.clone();
                let order_before = lfu_cache
                    .iter()
                    .filter(|(_, v)| **v % 2 == 0)
                    .map(|(k, v)| (*k, *v))
                    .collect::<Vec<_>>();

                lfu_cache.retain(|_k, v| *v % 2 == 0);

                let order_after = lfu_cache.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();

                assert!(lfu_cache.len() <= len_before);

                assert_eq!(
                    order_after, order_before,
                    "Retain operation changed the order unexpectedly from {:#?} to {:#?}",
                    all_before, lfu_cache,
                );

                assert!(
                    order_after.iter().all(|(_, v)| *v % 2 == 0),
                    "Retain operation left odd values"
                );

                let len_before = lru_cache.len();

                let order_before = lru_cache
                    .iter()
                    .filter(|(_, v)| **v % 2 == 0)
                    .map(|(k, v)| (*k, *v))
                    .collect::<Vec<_>>();

                lru_cache.retain(|_k, v| *v % 2 == 0);

                assert!(lru_cache.len() <= len_before);
                assert_eq!(
                    lru_cache.len(),
                    order_before.len(),
                    "Retain operation changed the length unexpectedly"
                );

                assert!(
                    lru_cache.iter().all(|(_, v)| *v % 2 == 0),
                    "Retain operation left odd values"
                );

                let len_before = mru_cache.len();

                let order_before = mru_cache
                    .iter()
                    .filter(|(_, v)| **v % 2 == 0)
                    .map(|(k, v)| (*k, *v))
                    .collect::<Vec<_>>();

                mru_cache.retain(|_k, v| *v % 2 == 0);

                assert!(mru_cache.len() <= len_before);
                assert_eq!(
                    mru_cache.len(),
                    order_before.len(),
                    "Retain operation changed the length unexpectedly"
                );

                assert!(
                    mru_cache.iter().all(|(_, v)| *v % 2 == 0),
                    "Retain operation left odd values"
                );
            }

            CacheOperation::Iter => {
                let lfu_items: Vec<_> = lfu_cache.iter().map(|(k, v)| (*k, *v)).collect();
                let lru_items: Vec<_> = lru_cache.iter().map(|(k, v)| (*k, *v)).collect();
                let mru_items: Vec<_> = mru_cache.iter().map(|(k, v)| (*k, *v)).collect();

                assert_eq!(lfu_items.len(), lfu_cache.len());
                assert_eq!(lru_items.len(), lru_cache.len());
                assert_eq!(mru_items.len(), mru_cache.len());

                for (key, value) in &lfu_items {
                    assert_eq!(lfu_cache.peek(key), Some(value));
                }
                for (key, value) in &lru_items {
                    assert_eq!(lru_cache.peek(key), Some(value));
                }
                for (key, value) in &mru_items {
                    assert_eq!(mru_cache.peek(key), Some(value));
                }

                let mut lfu_popped = Vec::new();
                while let Some((key, value)) = lfu_cache.pop() {
                    lfu_popped.push((key, value));
                }

                assert_eq!(lfu_popped, lfu_items);

                let mut lru_popped = Vec::new();
                while let Some((key, value)) = lru_cache.pop() {
                    lru_popped.push((key, value));
                }

                assert_eq!(lru_popped, lru_items);

                let mut mru_popped = Vec::new();
                while let Some((key, value)) = mru_cache.pop() {
                    mru_popped.push((key, value));
                }

                assert_eq!(mru_popped, mru_items);
            }
        }

        assert!(lfu_cache.len() <= lfu_cache.capacity());
        assert_eq!(lfu_cache.capacity(), initial_capacity);
        assert_eq!(lfu_cache.is_empty(), lfu_cache.len() == 0);

        if !lfu_cache.is_empty() {
            assert!(lfu_cache.tail().is_some());
        } else {
            assert_eq!(lfu_cache.tail(), None);
        }
    }
});
