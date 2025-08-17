#[derive(Debug)]
pub enum CacheOperation {
    Insert(u16, u16),
    Get(u16),
    Peek(u16),
    Remove(u16),
    Pop,
    Clear,
    GetOrInsertWith(u16, u16),
    Retain,
    Iter,
    PeekMut(u16, u16),
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
            9 => Ok(CacheOperation::PeekMut(u.arbitrary()?, u.arbitrary()?)),
            _ => unreachable!(),
        }
    }
}
