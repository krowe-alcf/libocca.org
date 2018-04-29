# Add Vectors

!>
    OKL is still transitioning from **v0.2** &nbsp;&rarr;&nbsp; **v1.0** (ETA is end of May '18)
    <br/>
    Examples are being written for **v1.0** which won't work at the moment, but linked source code will work
    <br/>
    [Check the v0.2 &nbsp;&rarr;&nbsp; v1.0 guide to see how to make the v1.0 examples work in the current OCCA setup](/guide/okl/v1)
    <br/>
    **Sorry for the inconvenience!**

**Source Code**
- [main.cpp](https://github.com/libocca/occa/blob/master/examples/1_add_vectors/cpp/main.cpp)
- [addVectors.okl](https://github.com/libocca/occa/blob/master/examples/1_add_vectors/cpp/addVectors.okl)

---

Our _'Hello World!'_ example is adding two vectors in parallel.
We'll walk through parts of the example explaining the parts and logic behind each part.

## Initializing Host Data

We'll start off by allocating and initializing our vectors in the host

```cpp
int entries = 5;

float *a  = new float[entries];
float *b  = new float[entries];
float *ab = new float[entries];

for (int i = 0; i < entries; ++i) {
  a[i]  = i;
  b[i]  = 1 - i;
  ab[i] = 0;
}
```

We will sum vectors `a` and `b` into the vector `ab`

## Initializing Device

We'll initialize our device by giving it the backend we wish to use along with information required by that backend

::: tabs backend

- Serial

    ```cpp
    occa::device device("mode: 'Serial'");
    ```

- OpenMP

    ```cpp
    occa::device device("mode: 'OpenMP', threads: 4");
    ```

- Threads

    ```cpp
    occa::device device("mode: 'Threads', threads: 4, pinnedCores: [0, 1, 2, 3]");
    ```

- OpenCL

    ```cpp
    occa::device device("mode: 'OpenCL', deviceID: 0, platformID: 0");
    ```

- CUDA

    ```cpp
    occa::device device("mode: 'CUDA', deviceID: 0");
    ```

:::

## Initializing Device Data

We need to allocate device memory which we'll use to add the vectors on the device

```cpp
occa::memory o_a  = device.malloc(entries * sizeof(float));
occa::memory o_b  = device.malloc(entries * sizeof(float));
occa::memory o_ab = device.malloc(entries * sizeof(float));
```

We need to initialize `o_a` and `o_b` with the host's `a` and `b` vectors

```cpp
o_a.copyFrom(a);
o_b.copyFrom(b);
```

Note we don't initialize `o_ab` since it'll be initialized when we add `o_a` and `o_b` in the device

## Build and Run Kernel

> A _kernel_ is a function on a device

We now create a kernel in the device by giving it the filename where the kernel lives, along with the kernel name.
We'll show and explain the [addVectors.okl](/examples/add-vectors?id=addvectorsokl) source code in the end.

```cpp
occa::kernel addVectors = device.buildKernel("addVectors.okl",
                                             "addVectors");
```

We now launch the `addVectors` kernel as if it were a regular function

```cpp
addVectors(entries, o_a, o_b, o_ab);
```

?> Note that not only do we pass `occa::memory` objects, but we can also pass primitive types such as `int`, `float`, and `double`

## Copy Data to Host

To check if we added the vectors properly, we'll need to copy over the results in `o_ab` to the host

```cpp
o_ab.copyTo(ab);
```

## Cleaning Up

Let's make sure we got the right results

```cpp
for (int i = 0; i < entries; ++i) {
  if (ab[i] != (a[i] + b[i])) {
    throw 1;
  }
}
```

Lastly we free our host memory

```cpp
delete [] a;
delete [] b;
delete [] ab;
```

?> All `occa` objects use reference counting to auto-free anything allocated!

## addVectors.okl

```okl
@kernel void addVectors(const int entries,
                        const float *a,
                        const float *b,
                        float *ab) {
  for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {
    if (i < entries) {
      ab[i] = a[i] + b[i];
    }
  }
}
```

We annotate kernels with the `@kernel` attribute and parallel loops with `@outer` and `@inner`

We use the `@tile` attribute as shorthand notation for

```okl
for (int offset = 0; offset < entries; offset += 16; @outer) {
  for (int i = offset; i < (offset + 16); ++i; @inner) {
    if (i < entries) {
      ab[i] = a[i] + b[i];
    }
  }
}
```

Note in both cases we have the conditional check `if (i < entries)`.
Since we tile by `16`, we need to make sure the iterations don't go out of bounds.
