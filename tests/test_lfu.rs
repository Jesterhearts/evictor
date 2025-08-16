use std::num::NonZeroUsize;

use evictor::Lfu;

#[test]
fn test_lfu_new_empty() {
    let cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 3);
    assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_lfu_insert_single() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string())]
    );
}

#[test]
fn test_lfu_insert_multiple() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.len(), 3);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string())
        ]
    );
}

#[test]
fn test_lfu_insert_overflow() {
    let mut cache = Lfu::new(NonZeroUsize::new(2).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (3, "three".to_string())]
    );
}

#[test]
fn test_lfu_get_existing() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.get(&2), Some(&"two".to_string()));
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (1, "one".to_string()),
            (3, "three".to_string()),
            (2, "two".to_string())
        ]
    );
}

#[test]
fn test_lfu_get_nonexistent() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    assert_eq!(cache.get(&2), None);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string())]
    );
}

#[test]
fn test_lfu_get_mut() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    if let Some(value) = cache.get_mut(&1) {
        *value = "ONE".to_string();
    }
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (1, "ONE".to_string())]
    );
}

#[test]
fn test_lfu_peek() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    assert_eq!(cache.peek(&1), Some(&"one".to_string()));
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string()), (2, "two".to_string())]
    );
}

#[test]
fn test_lfu_peek_mut() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    if let Some(mut entry) = cache.peek_mut(&1) {
        *entry = "ONE".to_string();
    }
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (1, "ONE".to_string())]
    );
}

#[test]
fn test_lfu_contains_key() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    assert!(cache.contains_key(&1));
    assert!(!cache.contains_key(&2));
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string())]
    );
}

#[test]
fn test_lfu_remove_existing() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.remove(&2), Some("two".to_string()));
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string()), (3, "three".to_string())]
    );
}

#[test]
fn test_lfu_remove_nonexistent() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    assert_eq!(cache.remove(&2), None);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string())]
    );
}

#[test]
fn test_lfu_pop() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.pop(), Some((1, "one".to_string())));
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (3, "three".to_string())]
    );
}

#[test]
fn test_lfu_pop_empty() {
    let mut cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    assert_eq!(cache.pop(), None);
    assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_lfu_tail() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.tail(), Some((&1, &"one".to_string())));
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string())
        ]
    );
}

#[test]
fn test_lfu_tail_empty() {
    let cache = Lfu::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    assert_eq!(cache.tail(), None);
    assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_lfu_clear() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_lfu_get_or_insert_with_new() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    let value = cache.get_or_insert_with(2, |&key| format!("value_{}", key));
    assert_eq!(value, &"value_2".to_string());
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string()), (2, "value_2".to_string())]
    );
}

#[test]
fn test_lfu_get_or_insert_with_existing() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    let value = cache.get_or_insert_with(1, |&key| format!("value_{}", key));
    assert_eq!(value, &"one".to_string());
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (1, "one".to_string())]
    );
}

#[test]
fn test_lfu_extend() {
    let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());
    cache.insert(1, "one".to_string());
    cache.extend(vec![(2, "two".to_string()), (3, "three".to_string())]);
    assert_eq!(cache.len(), 3);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (1, "one".to_string()),
            (2, "two".to_string()),
            (3, "three".to_string())
        ]
    );
}

#[test]
fn test_lfu_retain() {
    let mut cache = Lfu::new(NonZeroUsize::new(5).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    cache.insert(4, "four".to_string());
    cache.retain(|&key, _| key % 2 == 0);
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (4, "four".to_string())]
    );
}

#[test]
fn test_lfu_frequency_ordering() {
    let mut cache = Lfu::new(NonZeroUsize::new(4).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    cache.insert(4, "four".to_string());
    cache.get(&1);
    cache.get(&1);
    cache.get(&2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (3, "three".to_string()),
            (4, "four".to_string()),
            (2, "two".to_string()),
            (1, "one".to_string())
        ]
    );
}

#[test]
fn test_lfu_eviction_by_frequency() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    cache.get(&1);
    cache.get(&2);
    cache.insert(4, "four".to_string());
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (4, "four".to_string()),
            (1, "one".to_string()),
            (2, "two".to_string())
        ]
    );
}

#[test]
fn test_lfu_update_existing() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(1, "ONE".to_string());
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (1, "ONE".to_string())]
    );
}

#[test]
fn test_lfu_iter_into_iter() {
    let mut cache = Lfu::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());

    let iter = cache
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect::<Vec<_>>();
    let into_iter: Vec<_> = cache.into_iter().collect();

    assert_eq!(iter, into_iter);
}
