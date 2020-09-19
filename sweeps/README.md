Design Space Exploration with SMAUG
===================================

To enable fast design sweeps of deep learning systems built on SMAUG, we use
this sweep generator to lessen the burden on the user of manually writing
simulation configuration files.

# Generating and running sweeps #

A quick example of sweep the 4-FC minerva model:

```bash
./main.py --model=minerva --params=input.json --output-dir=output --run-points
```

This will generate sweeps under the `output` folder and run all of them (if
you only want to generate the configuration files without running any, remove
`--run-points`). The swept parameters are specified in a JSON file and past to
the generator using `--params`. In the case of `input.json`, we sweep the number
of accelerators of 1, 2, 4 and 8, thus generating 4 sweeps. You can change the
parameter values to generate designs.

The sweep generator looks for model/parameter files under `experiments/models`.
You can sweep other models created there. For example, `--model=lenet5` will use
the 5-layer LeNet5 model.

# Adding a new parameter #

Adding a new parameter to the sweep generator is fairly easy. All you need to do
is inheriting the `BaseParam` class (in `params.py`, in which you need to
implement two functions: `apply()` and `default_value()`. The former applies the
parameter sweep to the corresponding templated configuration files and the
latter gives a default value if no swept values are specified by the user. Take
the example of `L2SizeParam`, its `apply()` simply replaces the `l2_size`
keyword with the sweep value in the template file `configs/run.sh`, and
`default_value` returns 4MB as the default L2 cache size. You also need to add
a new item to the dict named `param_types` in `sweeper.py`, which maps a
parameter's name to its type. In this case, we add `"l2_size": L2SizeParam`, so
that you can use `l2_size` in a sweep file:

```json
{
    "l2_size": [1048576, 2097152, 4194304]
}
```

The above will sweep three L2 cache sizes.
