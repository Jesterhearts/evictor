#[derive(Clone, Copy)]
pub enum CacheOperation {
    Insert(u8, u8),
    Get(u8),
    Peek(u8),
    Remove(u8),
    Pop,
    Clear,
    GetOrInsertWith(u8, u8),
    Retain,
    Iter,
    PeekMut(u8, u8),
}

impl std::fmt::Debug for CacheOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheOperation::Insert(k, v) => write!(f, "cache.insert({k}, {v})"),
            CacheOperation::Get(k) => write!(f, "cache.get(&{k})"),
            CacheOperation::Peek(k) => write!(f, "cache.peek(&{k})"),
            CacheOperation::Remove(k) => write!(f, "cache.remove(&{k})"),
            CacheOperation::Pop => write!(f, "cache.pop()"),
            CacheOperation::Clear => write!(f, "cache.clear()"),
            CacheOperation::GetOrInsertWith(k, v) => {
                write!(f, "cache.get_or_insert_with({k}, |_| {v})")
            }
            CacheOperation::Retain => write!(f, "cache.retain(|k, _| k % 2 == 0)"),
            CacheOperation::Iter => write!(f, "cache.iter()"),
            CacheOperation::PeekMut(k, v) => write!(f, "cache.peek_mut(&{k}, {v})"),
        }
    }
}

impl<'a> arbitrary::Arbitrary<'a> for CacheOperation {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        match u.int_in_range(0..=9)? {
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
            9 => Ok(CacheOperation::PeekMut(u.arbitrary()?, u.arbitrary()?)),
            _ => unreachable!(),
        }
    }
}
