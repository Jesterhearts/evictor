#![no_main]
use std::num::NonZeroUsize;

use evictor::SIEVE;
use fuzz_lib::CacheOperation;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (u8, Vec<CacheOperation>)| {
    let _ = std::panic::take_hook();
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        default_hook(p);
    }));

    let (size, operations) = data;
    let size = size.max(1);

    let mut cache = SIEVE::<u8, u8>::new(NonZeroUsize::new(size as usize).unwrap());

    for operation in operations {
        assert!(
            cache.len() <= size as usize,
            "Cache size exceeded: {cache:#?}",
        );
        assert_eq!(
            cache.capacity(),
            size as usize,
            "Cache capacity altered: {cache:#?}",
        );
        cache.debug_validate();

        match operation {
            CacheOperation::Insert(k, v) => {
                let len = cache.len();
                let contains_key = cache.contains_key(&k);
                let after = *cache.insert(k, v);
                assert!(
                    contains_key || cache.len() > len || len == cache.capacity(),
                    "Insert operation failed to increase cache size: {} < {len} {k} {v} {cache:#?}",
                    cache.len()
                );

                assert!(
                    cache.contains_key(&k),
                    "Cache does not contain key: {k} {v} {cache:#?}",
                );
                assert!(
                    cache.iter().any(|(key, _)| key == &k),
                    "Cache does not contain key in iterator: {k} {v} {cache:#?}",
                );
                assert_eq!(
                    after, v,
                    "Insert operation failed for key: {k} {v} {cache:#?}",
                );
            }
            CacheOperation::Get(k) => {
                let contains_key = cache.contains_key(&k);
                let value = cache.get(&k).copied();
                assert!(
                    !contains_key || value.is_some(),
                    "Get operation failed for key: {k} {cache:#?}",
                );
                assert_eq!(
                    contains_key,
                    cache.contains_key(&k),
                    "Cache contains_key mismatch for {k}: {cache:#?}"
                );
                assert_eq!(
                    contains_key,
                    cache.iter().any(|(key, _)| key == &k),
                    "Cache does not contain key in iterator after get: {k} {cache:#?}",
                );
            }
            CacheOperation::Peek(k) => {
                let order_before = cache.iter().map(|(key, _)| *key).collect::<Vec<_>>();
                let contains_key = cache.contains_key(&k);
                let value = cache.peek(&k).copied();
                let order_after = cache.iter().map(|(key, _)| *key).collect::<Vec<_>>();
                assert!(
                    !contains_key || value.is_some(),
                    "Peek operation failed for key: {k} {cache:#?}",
                );
                assert_eq!(
                    contains_key,
                    cache.contains_key(&k),
                    "Cache contains_key mismatch for {k}: {cache:#?}"
                );
                assert_eq!(
                    order_before, order_after,
                    "Peek operation altered cache order for key: {k} {cache:#?}"
                );
            }
            CacheOperation::Remove(k) => {
                let len = cache.len();
                let contains_key = cache.contains_key(&k);
                let removed = cache.remove(&k);
                assert!(
                    !contains_key || removed.is_some(),
                    "Remove operation failed to remove key: {k} {cache:#?}",
                );
                assert!(
                    !contains_key || cache.len() < len,
                    "Remove operation did not decrease cache size: {k} {cache:#?}",
                );
                assert!(
                    !cache.contains_key(&k),
                    "Cache still contains key after remove: {k} {cache:#?}",
                );
                assert!(
                    cache.iter().all(|(key, _)| key != &k),
                    "Cache still contains key in iterator after remove: {k} {cache:#?}",
                )
            }
            CacheOperation::Pop => {
                let cache_before = cache.clone();
                let head = cache.iter().next().map(|(k, v)| (*k, *v));
                let popped = cache.pop();
                assert_eq!(
                    popped, head,
                    "Pop operation did not return expected value: {cache_before:#?} {cache:#?}",
                );
            }
            CacheOperation::Clear => {
                cache.clear();
                assert!(
                    cache.is_empty(),
                    "Clear operation did not empty cache: {cache:#?}",
                );
                assert_eq!(
                    cache.len(),
                    0,
                    "Cache length after clear is not zero: {cache:#?}",
                );
            }
            CacheOperation::GetOrInsertWith(k, v) => {
                let before = cache.get(&k).copied();
                let inserted = *cache.get_or_insert_with(k, |ik| {
                    assert_eq!(k, *ik, "Key mismatch in get_or_insert_with: {ik} != {k}");
                    v
                });
                assert!(
                    (before.is_none() && inserted == v) || before == Some(inserted),
                    "GetOrInsertWith operation failed for key: {k} {v} {cache:#?}",
                );
                assert!(
                    cache.contains_key(&k),
                    "Cache does not contain key after get_or_insert_with: {k} {v} {cache:#?}",
                );
            }
            CacheOperation::Retain => {
                let entries_before = cache.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
                cache.retain(|k, _| k % 2 == 0);
                let entries_after = cache.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
                assert!(
                    entries_after.iter().all(|(k, _)| k % 2 == 0),
                    "Retain operation did not filter correctly: {entries_after:#?}",
                );
                assert!(
                    entries_before.len() >= entries_after.len(),
                    "Retain operation increased cache size: {entries_before:#?} -> {entries_after:#?}",
                );
                let entries_before_filtered = entries_before
                    .into_iter()
                    .filter(|(k, _)| k % 2 == 0)
                    .collect::<Vec<_>>();
                assert_eq!(
                    entries_after, entries_before_filtered,
                    "Retain operation did not retain expected entries: {entries_after:#?} != {entries_before_filtered:#?} {cache:#?}",
                );
            }
            CacheOperation::Iter => {
                let mut count = 0;
                for (k, v) in cache.iter() {
                    assert!(
                        cache.contains_key(k),
                        "Iterator returned key not in cache: {k} {cache:#?}",
                    );
                    assert_eq!(
                        Some(*v),
                        cache.peek(k).copied(),
                        "Iterator returned value not matching cache: {k} {v} {cache:#?}",
                    );
                    count += 1;
                }
                assert_eq!(
                    count,
                    cache.len(),
                    "Iterator count does not match cache length: {cache:#?}",
                );
            }
            CacheOperation::PeekMut(k, v) => {
                let order_before = cache.iter().map(|(key, _)| *key).collect::<Vec<_>>();
                {
                    let value = cache.peek_mut(&k);
                    if let Some(v) = &value {
                        // Read the value without modifying it, use hint to avoid this touch getting
                        // optimized out
                        std::hint::black_box(v.value());
                    }
                }
                let order_after = cache.iter().map(|(key, _)| *key).collect::<Vec<_>>();
                assert_eq!(
                    order_before, order_after,
                    "PeekMut non-mutating operation altered cache order for key: {k} {cache:#?}",
                );
                {
                    let value = cache.peek_mut(&k);
                    if let Some(mut value) = value {
                        *value = v;
                    }
                }
                let after = cache.peek(&k).copied();
                assert_eq!(
                    after,
                    Some(v).filter(|_| cache.contains_key(&k)),
                    "PeekMut operation did not set value correctly for key: {k} {v} {cache:#?}",
                );
            }
        }
    }

    cache.debug_validate();
    assert!(
        cache.len() <= size as usize,
        "Cache size exceeded: {cache:#?}",
    );
    assert_eq!(
        cache.capacity(),
        size as usize,
        "Cache capacity altered: {cache:#?}",
    );
});
