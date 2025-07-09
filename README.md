# Replay

Replay is a Python-based framework designed to simulate the spread of infectious diseases over real-world human contact networks, using GPU acceleration for high performance at scale.

Built with CUDA 12.8 and Python 3.12.9, Replay is intended for researchers, epidemiologists, and public health professionals looking for a practical and scalable approach to modeling transmission dynamics using empirical data.

## Background

Understanding how diseases spread through dynamic human contact networks is essential for effective public health planning. Most traditional epidemic models assume simplified or static interaction patterns, which often don’t reflect the complexity of real-world contact behavior.

With growing access to mobility and proximity data, there’s a clear opportunity to build more accurate, data-driven models. However, working with time-resolved contact data can quickly become computationally expensive.

## What Replay Does

Replay addresses this by:

- Converting timestamped contact data into duration-weighted contact graphs  
- Using sparse matrix operations to compute exposure at the individual level  
- Running massive numbers of stochastic simulations in parallel via CUSPARSE  

Rather than approximating, Replay “replays” the actual contact patterns recorded in empirical datasets. This allows for realistic simulation of outbreaks, including localized spread and superspreader events, without hardcoded behavioral assumptions.

## Performance

Replay is optimized for speed and scale. On GPU, it’s capable of running thousands of Monte Carlo simulations per second, significantly outperforming CPU-based approaches, especially in high-density contact scenarios.

## Use Cases

- Public health planning  
- Hospital infection control  
- Real-time outbreak analysis  
- Research into contact-driven epidemic dynamics  

---

## 🔧 Build Walkthrough (Windows)

This section walks you through building Replay from source on **Windows** using **Visual Studio 2022**, **vcpkg**, and **Ninja**.

### ✅ Prerequisites

Make sure the following are installed and available in your system `PATH`:

- [x] [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)  
  – Include the **C++**, **CMake**, and **Windows SDK** components  
- [x] [CUDA Toolkit 12.8+](https://developer.nvidia.com/cuda-toolkit)  
- [x] Git  
- [x] Python 3.12  

---

### Step-by-Step Instructions

1. **Clone the repository and vcpkg submodule**  
   ```bat
   git clone https://github.com/your-username/replay.git
   cd replay
   git clone https://github.com/microsoft/vcpkg.git external/vcpkg
   cd external/vcpkg
   bootstrap-vcpkg.bat
   cd ../..

2. **Install dependencies with manifest mode**

    ```bat
    external\vcpkg\vcpkg install
    ```

3. **Create the build directory and configure with CMake**

    ```bat
    mkdir build
    cd build
    cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Debug ^
      -DCMAKE_TOOLCHAIN_FILE=../external/vcpkg/scripts/buildsystems/vcpkg.cmake
    ```

4. **Build the project**

    ```bat
    cmake --build .
    ```

    The executable will be generated in the `tool/build` directory as `replay.exe`.

---

### :repeat: Force a Clean Rebuild

```bat
rd /s /q build
mkdir build
cd build
cmake .. -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=Debug ^
  -DCMAKE_TOOLCHAIN_FILE=../external/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build .
```

---

## Usage

This section outlines the basic steps needed to begin using the tool.

### Setup Instructions

The user interface is just a fancy way to interact with the command line utility provided by the executable. As such, in order to run the UI, you will need to build or download the `replay.exe` binary and place it in the same directory as `run_sim_gui.py`.
Once this setup step has been completed you can launch the UI tool by running `py run_sim_gui.py`.

### Graph Format
ALL simulations take place on a temporal graph.
The graph **MUST** be provided in the following format:
```
NODE_LIST [NODE ID] [NODE ID] [NODE ID] ...
[NODE ID], [NODE ID], [START OF CONTACT BETWEEN NODES - UTC FORMAT], [END OF CONTACT BETWEEN NODES - UTC FORMAT];
[NODE ID], [NODE ID], [START OF CONTACT BETWEEN NODES - UTC FORMAT], [END OF CONTACT BETWEEN NODES - UTC FORMAT];
...
```
For example, the following graph describes a series of contacts between five individuals at a medical facility.
Note that each distinct contact is described by a seperate entry, even between the same pair of individuals.
```
NODE_LIST 55 63 65 22 23052832
55, 22, 2000-01-01T05:24:44Z, 2000-01-01T25:59:59Z;
22, 23052832, 2000-01-01T09:01:33Z, 2000-01-01T22:34:32Z;
55, 22, 2000-01-05T20:06:55Z, 2001-02-20T21:33:05Z;
```
In an actual graph file, there might be thousands of nodes and millions of recorded contacts. The main bottleneck is the number of simultaneous interactions - as this increases, VRAM requirements also increase. 

**IMPORTANT** - contacts are inherently undirected, which means that if node 1 comes into contact with node 2 for 10 minutes, node 2 also comes into contact with node 1 for 10 minutes.

**In other words, you should make sure that your data is not formatted like this, otherwise it will be double-counted:**
```
NODE_LIST 1 2
1, 2, 2000-01-01T00:00:00Z, 2000-01-02T00:00:00Z;
2, 1, 2000-01-01T00:00:00Z, 2000-01-02T00:00:00Z;
```

### Simulation Parameters
The following parameters can be adjusted as needed to produce a realistic epidemic simulation. These values can be easily configured by launching the user interface.
#### File Configuration
| Parameter | Display Name | Description |
|----------|--------------|-------------|
| `temporal_contact_file` | **Contact Network File** | Input file containing time-stamped contact data (required). |
| `--summary-out` | **Summary Output CSV** | CSV file logging SEIR state counts per time step. |
| `--node-state-out` | **Node-Level Output CSV** | CSV file recording each individual's state over time. |
| `--visualize` | **Visualize Summary** | Automatically generate a plot from the summary output. |


####  Simulation Configuration

| Parameter | Display Name | Description |
|-----------|--------------|-------------|
| `--use-cpu` | **Run on CPU** | Force simulation to run on CPU instead of GPU. **Much slower!** |
| `--cpu-threads` | **CPU Threads** | Number of CPU threads, if CPU simulation enabled. |
| `--M` | **Parallel Simulations** | Number of Monte Carlo simulations to run in parallel. |
| `--step-size` | **Step Size (seconds)** | Duration of each simulation step in seconds. |
| `--iterations` | **Number of Steps** | Total number of time steps in the simulation. |

---

#### Epidemiological Parameters

| Parameter | Display Name | Description |
|-----------|--------------|-------------|
| `--initial-infected` | **Initial Infection Probability** | Probability that each individual starts infected. |
| `--infect-prob` | **Transmission Probability** | Probability of infection per hour of exposure. |
| `--exposed-duration` | **Incubation Period (seconds)** | Time before an exposed person becomes infectious. |
| `--infectious-duration` | **Infectious Period (seconds)** | Duration a person remains infectious. |
| `--resistant-duration` | **Resistance Duration (seconds)** | Time a recovered person is immune before returning to susceptible. |

---

#### Time Configuration

| Parameter | Display Name | Description |
|-----------|--------------|-------------|
| `--start-date` | **Start Date (UTC)** | Start date for the simulation in UTC format. Make sure it aligns with your data, otherwise the infection will not spread!|
| `--start-time` | **Start Time (UTC)** | Start time (HH:MM:SS) to pair with the start date.|
| `--time-step` | **Time Step Size (seconds)** | Duration of each contact network slice. Rapidly changing network structures will need smaller values.|

---

### For Maintainers

**How to update the vcpkg baseline**

```bat
git clone https://github.com/microsoft/vcpkg.git external/vcpkg
cd external/vcpkg
bootstrap-vcpkg.bat
vcpkg x-update-baseline
cd ../..
```

> This updates the `builtin-baseline` field in `vcpkg.json` to match the latest commit from the vcpkg registry.  
> Be sure to **commit the updated `vcpkg.json`** so others build with the same dependency versions.

