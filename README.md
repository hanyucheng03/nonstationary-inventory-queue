# Replication Package

Replication code for:

**Optimal Control of a Non-Stationary Inventory-Queue, with Applications to EV-Battery Swapping**

Hanyu Cheng and David D. Yao

arXiv:2609.35153

## Files

- `run_all.py` — master script. Runs all experiments in the order needed by the paper and checks that all manuscript figure files were generated.
- `parameters.csv` — compact summary of the main parameter settings used in each experiment.
- `requirements.txt` — Python dependencies.

### `code/`

- `utah_summer_primitive_experiment.py` — summer electricity-price experiment used in Figures 1–2.
- `utah_summer_primitive_experiment_lower_power.py` — lower-charging-power summer experiment used in Figure 2.
- `utah_winter_primitive_experiment.py` — winter electricity-price experiment used in Figure 3.
- `utah_winter_primitive_experiment_lower_power.py` — lower-charging-power winter experiment used in Figure 4.
- `robust_preemptive.py` — phase-type threshold experiment used in Figure 5.
- `ph_scv_objective.py` — baseline phase-type/SCV cost experiment used in Figure 6(a).
- `ph_scv_objective_counter.py` — higher-demand/higher-penalty phase-type/SCV experiment used in Figure 6(b).
- `ph_general_mean_compare.py` — bimodal, component-matched Erlang, and mean-matched exponential experiments used in Figures 7–8; it also writes the CSV files used by the final plotting script.
- `ex_general_scv_compare.py` — generates the final Figures 7–8 from the comparison CSV files. It is normally called through `run_all.py`.

### `results/`

All numerical outputs and manuscript figures are written here.

## How to run

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run everything:

```bash
python run_all.py
```

Run only one group of experiments:

```bash
python run_all.py --subset primitive
python run_all.py --subset preemptive
python run_all.py --subset ph-scv
python run_all.py --subset bimodal
```

Regenerate only the final Figures 7–8 from existing CSV files:

```bash
python run_all.py --subset figures
```

The full run can take several minutes, especially the phase-type/SCV experiments.
