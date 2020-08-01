SMAUG: Simulating Machine Learning Accelerators Using gem5-Aladdin
==================================================================

[![harvard-acc](https://circleci.com/gh/harvard-acc/smaug.svg?style=shield)](https://circleci.com/gh/harvard-acc/smaug)

SMAUG is a deep learning framework that enables end-to-end simulation of DL models
on custom SoCs with a variety of hardware accelerators. SMAUG is designed to
enable DNN researchers to rapidly evaluate different accelerator and SoC
designs and perform hardware-software co-design. Simulation is powered by the
[gem5-Aladdin](https://github.com/harvard-acc/gem5-aladdin) SoC simulator,
allowing users to easily write new hardware accelerators and integrate them
into SMAUG for testing and exploration.

SMAUG provides stable Python and C++ APIs, allowing users to work at varying
levels of abstraction. For example, researchers can:

* Focus on building and evaluating models by working with the high-level Python API.
* Focus on evaluating different SoC configurations through gem5.
* Evaluate new tiling strategies and configurations through th C++ tiling
  optimizers.
* Build new accelerators and integrate them into SMAUG.

If you are using SMAUG in research, we would appreciate a reference to:

Sam (Likun) Xi, Yuan Yao, Kshitij Bhardwaj, Paul Whatmough, Gu-Yeon Wei, and
David Brooks. 2020. SMAUG: End-to-End Full-Stack Simulation Infrastructure for
Deep Learning Workloads. J. ACM 0, 0, Article 0 (2020). [PDF to be linked].

# Installation #

SMAUG requires use of a Docker image, available on Docker Hub
[here](https://registry.hub.docker.com/repository/docker/xyzsam/smaug).
Users who cannot use Docker can follow the commands in the
[Dockerfile](https://github.com/harvard-acc/smaug/blob/master/docker/Dockerfile)
to set up a local environment. However, due to our limited resources, we can
only offer support to Docker users.

To install SMAUG, first install Docker, then pull the Docker image:

```bash
docker pull xyzsam/smaug:latest
```

Then, run the following command to create a Docker volume that hosts your
workspace and start the container. The local volume (aka `smaug-workspace`
below) will store all your source code, local changes, and build artifacts, so
that you can start/stop the Docker container without losing any of your work.

```bash
docker run -it --rm --mount source=smaug-workspace,target=/workspace xyzsam/smaug:latest
```

The Docker container already contains all the source code repositories you
need, but they are probably out of date.  You will need to update them. Go into
your /workspace directory and run the following commands:

```bash
cd gem5-aladdin && git pull origin master && git submodule update --init --recursive && cd ..
cd LLVM-Tracer && git pull origin master && cd ..
cd smaug && git pull origin master && git submodule update --init --recursive && cd ..
```

You are now ready to work with SMAUG. Read on to learn how to run your first
model.

# Running your first model #

# Resources #

