use std::num::NonZeroUsize;

use evictor::Lfu;

fn main() {
    let mut cache = Lfu::<i32, i32>::new(NonZeroUsize::new(300000).unwrap());
    for _ in 0..3000 {
        for i in 0..100000 {
            cache.insert(i, i);
        }
        for i in 0..100000 {
            std::hint::black_box(cache.remove(std::hint::black_box(&i)));
        }
    }
    std::hint::black_box(cache);
}
