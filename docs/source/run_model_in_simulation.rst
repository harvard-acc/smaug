Run a model in gem5-Aladdin simulation
======================================

Following the tutorial in :ref:`label_build_python_model`, now we are able to
create a DL model using the SMAUG Python APIs. In this tutorial, we will
proceed to introduce the steps of running a model in gem5-Aladdin simulation.
We strongly recommend using our `Docker image
<https://registry.hub.docker.com/r/xyzsam/smaug>`_ and this tutorial assumes we
are in the Docker image.

Build gem5-Aladdin
------------------

First, if you haven't built gem5-Aladdin (cloned in
:code:`/workspace/gem5-aladdin`), use the following command to build it.

.. code-block:: console

    python2.7 `which scons` build/X86/gem5.opt PROTOCOL=MESI_Two_Level_aladdin -j2

Here, the :code:`MESI_Two_Level_aladdin` is the coherence protocol we will use
in the gem5 Ruby memory model. Change :code:`-j` parameter to increase the
number of CPU threads to speed up the build, but keep in mind that you may run
out of memory before you run out of CPUs. Running out of memory or disk space
can cause mysterious build failures.

Generate dynamic trace for gem5-Aladdin simulation
--------------------------------------------------

gem5-Aladdin, which provides the simulation capabilities of estimating
performance/power/area for pre-RTL kernels via various SoCs, is a trace-driven
simulator. Thus, in SMAUG, in order to run a model in gem5-Aladdin, we need to
generate a dynamic trace for the kernels to be simulated as hardware blocks.

First, we need to build the SMAUG tracer, which is an instrumented binary that
will be executed to generate the dynamic trace.

.. code-block:: console

    make tracer -j4

After running the above command, we will get a binary under :code:`build/bin/`
named :code:`smaug-instrumented`. Then we will run the model created in
:ref:`label_build_python_model` using the following command:

.. code-block:: console

    build/bin/smaug-instrumented my_model_topo.pbtxt my_model_params.pb

While the tracer is producing the dynamic trace, we will see outputs such as the
following pair for each invocation of the hardware blocks::

   dynamic_trace_acc0.gz: Starting to log at inst = 0.
   dynamic_trace_acc0.gz: Stopping logging at inst 18199217.

A trace file named :code:`dynamic_trace_acc0.gz` is generated.

Note that for large models, this step can take a long time and the resulting
trace file can be quite large. Sampling should be used to solve this, which is
discussed in :ref:`label_apply_sampling`.

Create gem5-Aladdin configuraion files
---------------------------------------

gem5-Aladdin also requires two configuration files for running simulations. One
file specifies SoC-level parameters such as each accelerator's ID, cache or DMA
configurations, location of the corresponding dynamic trace file, and etc.  An
example can found be in :code:`experiments/sims/smv/minerva/gem5.cfg` (make
sure the submodules are initialized by running :code:`git submodule update
--init --recursive`). The other file gives the parameters to be used for
implementing the accelerator, such as parameters for applying loop unrolling
and scratchpad partition. An example configuration file of implementing the
hardware blocks in our `SMV` backend can be found in
:code:`experiments/sims/smv/smv-accel.cfg`.  If you don't make any changes to
the `SMV` backend, both of these configuration files are sufficient to simulate
any model run on `SMV`. You can change the parameters but it may lead to
unexpected accelerator performance characteristics especially if you change
:code:`smv-accel.cfg`, as the array partitioning/loop unrolling/loop pipelining
are all fine tuned to achieve a specific throughput.

We defer to the `gem5-Aladdin GitHub repo
<https://github.com/harvard-acc/gem5-aladdin>`_ and `gem5-Aladdin tutorial
<http://accelerator.eecs.harvard.edu/micro16tutorial/slides/micro2016-tutorial-gem5-aladdin.pptx>`_
for more details of writing these configuration files.

Run the first simulation
------------------------

Assuming we use configuration files in
:code:`experiments/sims/smv/tests/minerva/gem5.cfg`
and :code:`experiments/sims/smv/smv-accel.cfg`, let's create a folder for our
first simulation. It contains these files:

.. code-block:: console

    root@72e3f78202e0:test # ls
    dynamic_trace_acc0.gz  gem5.cfg  my_model_params.pb  my_model_topo.pbtxt smv-accel.cfg

Now we are ready to launch the simulation.

.. code-block:: console

    /workspace/gem5-aladdin/build/X86/gem5.opt \
      --debug-flags=Aladdin,HybridDatapath \
      --outdir=outputs \
      /workspace/gem5-aladdin/configs/aladdin/aladdin_se.py \
      --num-cpus=1 \
      --mem-size=4GB \
      --mem-type=LPDDR4_3200_2x16  \
      --cpu-clock=2.5GHz \
      --cpu-type=DerivO3CPU \
      --ruby \
      --access-backing-store \
      --l2_size=2097152 \
      --l2_assoc=16 \
      --cacheline_size=32 \
      --accel_cfg_file=gem5.cfg \
      --fast-forward=10000000000 \
      -c /workspace/smaug/build/bin/smaug \
      -o "my_model_topo.pbtxt my_model_params.pb --gem5 --debug-level=0"

This command runs our custom 3-level DNN model in gem5-Aladdin simulation.
gem5-Aladdin provides a wide range of SoC simulation choices, for instance,
here, the simulated SoC has an out-of-order CPU running at 2.5GHZ, a two-level
cache hierarchy with a 2MB, 16-way associative L2 cache and 32B cacheline size.

There are a few parameters that have special importance:

- :code:`--mem-size=4GB` will attempt to allocate an enitre 4GB block of
  memory. If the host machine is low on memory, this could fail.
- The :code:`--num-cpus=1` can be adjusted to allow for multithreading (see
  :ref:`label_multithreading` for discussion).
- The :code:`--fast-forward` parameter is used to speed up the simulation of
  the initialization phase, which uses a simplified CPU model in gem5. It is
  set to a very large cycle value to ensure that gem5 doesn't automatically
  switch to detailed CPU too early. In SMAUG, we use gem5's magic instruction
  :code:`m5_switch_cpu` to switch to the detailed CPU when the initialization
  is done.
- For the SMAUG commandline options, note that now we must pass :code:`--gem5`
  flag to enable gem5-Aladdin simulation.

After the simulation starts, we will see the output look like this::

    Model topology file: my_model_topo.pbtxt
    Model parameters file: my_model_params.pb
    Number of accelerators: 1
    info: Increasing stack size by one page.
    ======================================================
    Loading the network model...
    ======================================================
    ======================================================
    Summary of the network.
    ======================================================
    ____________________________________________________________________________________________
    Layer (type)                             Output shape                 Parameters
    ____________________________________________________________________________________________
    data_2 (Data)                            (10, 6272)                       0
    ____________________________________________________________________________________________
    data_1 (Data)                            (32, 3, 3, 1)                    0
    ____________________________________________________________________________________________
    data (Data)                              (1, 28, 28, 1)                   0
    ____________________________________________________________________________________________
    conv (Convolution3d)                     (1, 28, 28, 32)                 288
    ____________________________________________________________________________________________
    max_pool (MaxPooling)                    (1, 14, 14, 32)                  0
    ____________________________________________________________________________________________
    reorder (Reorder)                        (1, 6272)                        0
    ____________________________________________________________________________________________
    mat_mul (InnerProduct)                   (1, 10)                        62720
    ____________________________________________________________________________________________

This means the model has been successfully loaded in the simulation. Then we
will see the following right after the network summary::

    ======================================================
    Tiling operators of the network...
    ======================================================
    Tiling conv (Convolution3d).
    Tiling data (Data).
    Tiling data_1 (Data).
    Tiling data_2 (Data).
    Tiling mat_mul (InnerProduct).
    Tiling max_pool (MaxPooling).
    Tiling reorder (Reorder).

This shows that SMAUG is performing pre-tiling procedures for each operator -
read-only tensors such as weights can be tiled before the actual layer-by-layer
network execution. After this, we can see the simulation switches to use the OoO
CPU model::

    Switched CPUS @ tick 29955086000
    switching cpus

This means the initialization is done and SMAUG will start scheduling operators
of the model. We can see::

    ======================================================
    Scheduling operators of the network...
    ======================================================
    Scheduling data (Data).
    Scheduling data_1 (Data).
    Scheduling data_2 (Data).
    Scheduling conv (Convolution3d).

As the :code:`conv` operator needs to invoke the convolution engine, we begin
to see the Aladdin simulation logs. After each invocation of the hardware
block, results are printed::

    ===============================
           Aladdin Results
    ===============================
    Running : ./outputs/nnet_fwd
    Top level function: smv_conv3d_nhwc_vec_fxp
    Cycle : 89771 cycles
    Upsampled Cycle : 0 cycles
    Avg Power: 132.997 mW
    Idle FU Cycles: 24432 cycles
    Avg FU Power: 111.49 mW
    Avg FU Dynamic Power: 102.124 mW
    Avg FU leakage Power: 9.36592 mW
    Avg MEM Power: 21.5071 mW
    Avg MEM Dynamic Power: 2.5961 mW
    Avg MEM Leakage Power: 18.911 mW
    Total Area: 1.91503e+06 uM^2
    FU Area: 775140 uM^2
    MEM Area: 1.13989e+06 uM^2
    Num of Multipliers (32-bit): 37
    Num of Adders (32-bit): 138
    Num of Bit-wise Operators (32-bit): 12
    Num of Shifters (32-bit): 21
    Num of Registers (32-bit): 1798
    ===============================
          Aladdin Results
    ===============================

Congratulations! You just finished running the first SMAUG simulation. The
:code:`outputs` folder contains simulation stats generated by gem5-Aladdin.

How long the simulation will take depends on the machine. On my Intel Core i7
9850H, it takes about 10 minutes. For larger models, both the trace storage and
simulation time can become problematic, read on to see the solution.

.. _label_apply_sampling:

Apply sampling to reduce simulation time and trace storage
----------------------------------------------------------

To save simulation time and trace storage for large models, we can sample the
loop iterations in the oftentimes repetitive DL kernels. To enable sampling,
use the :code:`--sample-level` and :code:`--sample-num` parameters in SMAUG.
:code:`--sample-level` sets the sampling level, which can be one of the five
levels: :code:`no`, :code:`low`, :code:`medium`, :code:`high` and
:code:`very_high`. :code:`--sample-num` sets the number of iterations a sampled
loop will run.  A smaller value means more sampling will be applied. Details
about using the sampling parameters and the gem5-Aladdin sampling APIs are
discussed in `Sampling of accelerated kernels <doxygen_html/simulation.html>`_.

.. _label_multithreading:

Simulate with multiple threads
------------------------------

If the SoC has multiple CPUs (specified via the :code:`--num-cpus` parameter),
we can enable SMAUG's multithreading feature by passing the
:code:`--num-threads` parameter to it. See `Multithreading in gem5 SE mode
<doxygen_html/simulation.html>`_ for more details.

