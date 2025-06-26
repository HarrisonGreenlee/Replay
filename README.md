# Replay
Replay is a Python-based framework designed to simulate the spread of infectious diseases over real-world human contact networks, using GPU acceleration for high performance at scale.

Built with CUDA 12.8 and Python 3.12.9, Replay is intended for researchers, epidemiologists, and public health professionals looking for a practical and scalable approach to modeling transmission dynamics using empirical data.

# Background

Understanding how diseases spread through dynamic human contact networks is essential for effective public health planning. Most traditional epidemic models assume simplified or static interaction patterns, which often don’t reflect the complexity of real-world contact behavior.

With growing access to mobility and proximity data, there’s a clear opportunity to build more accurate, data-driven models. However, working with time-resolved contact data—especially across large populations—can quickly become computationally expensive.

# What Replay Does

Replay addresses this by:

    Converting timestamped contact data into duration-weighted contact graphs

    Leveraging sparse matrix operations to compute exposure at the individual level

    Running massive numbers of stochastic simulations in parallel on commodity GPUs

Rather than approximating, Replay “replays” the actual contact patterns recorded in empirical datasets. This allows for realistic simulation of outbreaks, including localized spread and superspreader events, without hardcoded behavioral assumptions.

# Performance

Replay is optimized for speed and scale. On GPU, it’s capable of running thousands of Monte Carlo simulations per second, significantly outperforming CPU-based approaches, especially in high-density contact scenarios.


# Use Cases

    Public health planning

    Hospital infection control

    Real-time outbreak analysis

    Research into contact-driven epidemic dynamics

All code and benchmarks are available in this repository. Contributions, feedback, and extensions are welcome.
