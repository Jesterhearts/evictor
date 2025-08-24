use std::{
    hint::black_box,
    num::NonZeroUsize,
};

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
#[cfg(feature = "rand")]
use evictor::Random;
use evictor::{
    Fifo,
    Lfu,
    Lifo,
    Lru,
    Mru,
    Sieve,
};

macro_rules! bench_set {
    ($bench_set:ident, $cache:ident) => {
        mod $bench_set {
            use super::*;

            pub fn bench_put_update(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_put_update"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    for i in 0..10000 {
                        cache.insert(i, i);
                    }
                    b.iter(|| {
                        for i in 0..10000 {
                            black_box(cache.insert(i, i));
                        }
                    });
                });
                group.finish();
            }

            pub fn bench_put_insert(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_put_insert"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    b.iter(|| {
                        for i in 0..10000 {
                            black_box(cache.insert(i, i));
                        }
                    });
                });
                group.finish();
            }

            pub fn bench_get(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_get"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    for i in 0..10000 {
                        cache.insert(i, i);
                    }
                    b.iter(|| {
                        for i in 0..10000 {
                            black_box(cache.get(&i));
                        }
                    });
                });
                group.finish();
            }

            pub fn bench_remove(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_remove"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    for i in 0..10000 {
                        cache.insert(i, i);
                    }
                    b.iter(|| {
                        for i in 0..10000 {
                            black_box(cache.remove(&i));
                        }
                    });
                });
                group.finish();
            }

            pub fn bench_peek(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_peek"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    for i in 0..10000 {
                        cache.insert(i, i);
                    }
                    b.iter(|| {
                        for i in 0..10000 {
                            black_box(cache.peek(&i));
                        }
                    });
                });
                group.finish();
            }

            pub fn bench_get_not_found(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_get_not_found"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    for i in 0..10000 {
                        cache.insert(i, i);
                    }
                    b.iter(|| {
                        for i in 10000..20000 {
                            black_box(cache.get(&i));
                        }
                    });
                });
                group.finish();
            }

            pub fn bench_peek_not_found(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_peek_not_found"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    for i in 0..10000 {
                        cache.insert(i, i);
                    }
                    b.iter(|| {
                        for i in 10000..20000 {
                            black_box(cache.peek(&i));
                        }
                    });
                });
                group.finish();
            }

            pub fn bench_remove_not_found(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_remove_not_found"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    for i in 0..10000 {
                        cache.insert(i, i);
                    }
                    b.iter(|| {
                        for i in 10000..20000 {
                            black_box(cache.remove(&i));
                        }
                    });
                });
                group.finish();
            }

            pub fn bench_evict(c: &mut Criterion) {
                let mut group = c.benchmark_group(concat!(stringify!($cache), "_evict"));
                group.bench_function(criterion::BenchmarkId::from_parameter(10000), |b| {
                    let mut cache = $cache::new(NonZeroUsize::new(10000).unwrap());
                    for i in 0..10000 {
                        cache.insert(i, i);
                    }
                    b.iter(|| {
                        for i in 10000..20000 {
                            black_box(cache.insert(i, i));
                        }
                    });
                });
                group.finish();
            }
        }

        criterion_group!(
            $bench_set,
            $bench_set::bench_put_update,
            $bench_set::bench_put_insert,
            $bench_set::bench_get,
            $bench_set::bench_remove,
            $bench_set::bench_peek,
            $bench_set::bench_get_not_found,
            $bench_set::bench_peek_not_found,
            $bench_set::bench_remove_not_found,
            $bench_set::bench_evict,
        );
    };
}

bench_set!(lru, Lru);
bench_set!(mru, Mru);
bench_set!(lfu, Lfu);
bench_set!(sieve, Sieve);
bench_set!(fifo, Fifo);
bench_set!(lifo, Lifo);

#[cfg(not(feature = "rand"))]
criterion_main!(lru, lfu, mru, sieve, fifo, lifo);

#[cfg(feature = "rand")]
bench_set!(random, Random);
#[cfg(feature = "rand")]
criterion_main!(lru, mru, lfu, sieve, fifo, lifo, random);
