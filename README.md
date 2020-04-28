# p-SIR
## Particle dynamics for studying the SIR model

Scripts and programs that use particle collisions simulations to study the SIR model (and its extensions).

### Components

- [`dynamo/`](dynamo): Python scripts for preparing input files for, and reformatting output files from, the [DynamO](http://dynamomd.org) particle simulator.

- [`c-SIR/`](c-SIR): C-code for running the SIR model (or extensions of it) over a collisions list file, as produced by DynamO and reformatted by `dynamo/convert.py`.

- [`analysis/`](analysis): Python scripts for post-processing and visualizing the output from `c-SIR`.
