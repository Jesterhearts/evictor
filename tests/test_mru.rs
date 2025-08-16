use std::num::NonZeroUsize;

use evictor::Mru;

#[test]
fn test_mru_new_empty() {
    let cache = Mru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 3);
    assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_mru_insert_single() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string())]
    );
}

#[test]
fn test_mru_insert_multiple() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.len(), 3);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (3, "three".to_string()),
            (2, "two".to_string()),
            (1, "one".to_string())
        ]
    );
}

#[test]
fn test_mru_insert_overflow() {
    let mut cache = Mru::new(NonZeroUsize::new(2).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(3, "three".to_string()), (1, "one".to_string())]
    );
}

#[test]
fn test_mru_get_existing() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.get(&2), Some(&"two".to_string()));
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (2, "two".to_string()),
            (3, "three".to_string()),
            (1, "one".to_string())
        ]
    );
}

#[test]
fn test_mru_get_nonexistent() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    assert_eq!(cache.get(&2), None);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string())]
    );
}

#[test]
fn test_mru_get_mut() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    if let Some(value) = cache.get_mut(&1) {
        *value = "ONE".to_string();
    }
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "ONE".to_string()), (2, "two".to_string())]
    );
}

#[test]
fn test_mru_peek() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    assert_eq!(cache.peek(&1), Some(&"one".to_string()));
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (1, "one".to_string())]
    );
}

#[test]
fn test_mru_peek_mut() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    if let Some(mut entry) = cache.peek_mut(&1) {
        *entry = "ONE".to_string();
    }
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "ONE".to_string()), (2, "two".to_string())]
    );
}

#[test]
fn test_mru_contains_key() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    assert!(cache.contains_key(&1));
    assert!(!cache.contains_key(&2));
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string())]
    );
}

#[test]
fn test_mru_remove_existing() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.remove(&2), Some("two".to_string()));
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(3, "three".to_string()), (1, "one".to_string())]
    );
}

#[test]
fn test_mru_remove_nonexistent() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    assert_eq!(cache.remove(&2), None);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string())]
    );
}

#[test]
fn test_mru_pop() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.pop(), Some((3, "three".to_string())));
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "two".to_string()), (1, "one".to_string())]
    );
}

#[test]
fn test_mru_pop_empty() {
    let mut cache = Mru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    assert_eq!(cache.pop(), None);
    assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_mru_tail() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    assert_eq!(cache.tail(), Some((&3, &"three".to_string())));
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (3, "three".to_string()),
            (2, "two".to_string()),
            (1, "one".to_string())
        ]
    );
}

#[test]
fn test_mru_tail_empty() {
    let cache = Mru::<i32, String>::new(NonZeroUsize::new(3).unwrap());
    assert_eq!(cache.tail(), None);
    assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_mru_clear() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.into_iter().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_mru_get_or_insert_with_new() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    let value = cache.get_or_insert_with(2, |&key| format!("value_{}", key));
    assert_eq!(value, &"value_2".to_string());
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(2, "value_2".to_string()), (1, "one".to_string())]
    );
}

#[test]
fn test_mru_get_or_insert_with_existing() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    let value = cache.get_or_insert_with(1, |&key| format!("value_{}", key));
    assert_eq!(value, &"one".to_string());
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "one".to_string()), (2, "two".to_string())]
    );
}

#[test]
fn test_mru_extend() {
    let mut cache = Mru::new(NonZeroUsize::new(4).unwrap());
    cache.insert(1, "one".to_string());
    cache.extend(vec![(2, "two".to_string()), (3, "three".to_string())]);
    assert_eq!(cache.len(), 3);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (3, "three".to_string()),
            (2, "two".to_string()),
            (1, "one".to_string())
        ]
    );
}

#[test]
fn test_mru_retain() {
    let mut cache = Mru::new(NonZeroUsize::new(5).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    cache.insert(4, "four".to_string());
    cache.retain(|&key, _| key % 2 == 0);
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(4, "four".to_string()), (2, "two".to_string())]
    );
}

#[test]
fn test_mru_access_pattern() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(3, "three".to_string());
    cache.get(&1);
    cache.insert(4, "four".to_string());
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![
            (4, "four".to_string()),
            (3, "three".to_string()),
            (2, "two".to_string())
        ]
    );
}

#[test]
fn test_mru_update_existing() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
    cache.insert(1, "one".to_string());
    cache.insert(2, "two".to_string());
    cache.insert(1, "ONE".to_string());
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache.into_iter().collect::<Vec<_>>(),
        vec![(1, "ONE".to_string()), (2, "two".to_string())]
    );
}

#[test]
fn test_mru_iter_into_iter() {
    let mut cache = Mru::new(NonZeroUsize::new(3).unwrap());
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
