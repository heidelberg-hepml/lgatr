# Benchmarks

Speed benchmark for the `equi_linear` and `geometric_product` primitives. Reports forward and
backward times for `dense` and `sparse`, plus optional `+compile` legs. Speedups are vs
`dense`.

## Running

```bash
python benchmarks/bench_primitives.py [options]
```

Options:

- `--bench {linear,gp,both}` (default `both`)
- `--compile` — also run `dense+compile` and `sparse+compile`
- `--cpu` — force CPU even if CUDA is available
