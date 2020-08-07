# p-SEIQR
## Particle dynamics for studying the SEIQR model

Scripts and programs that use particle collisions simulations to study extensions of the S(E)IR model.

### Components

- [`dynamo/`](dynamo): Python scripts for preparing input files for, and reformatting output files from, the [DynamO](http://dynamomd.org) particle simulator.

- [`c-SIR/`](c-SIR): C-code for running the SIR model (or extensions of it) over a collisions list file, as produced by DynamO and reformatted by `dynamo/convert.py`.

- [`run/`](run): Python scripts that run `c-SIR`.

- [`analysis/`](analysis): Python scripts for post-processing and visualizing the output from `c-SIR`.

