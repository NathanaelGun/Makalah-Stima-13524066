# Theme Park Queue Optimization using Sequential A* Search

This repository contains a Python implementation of a spatiotemporal routing optimization algorithm that adapts the **A\* search algorithm** to solve a time-dependent, asymmetric variant of the Traveling Salesperson Problem (TSP) for theme park itinerary routing.

The code simulates and evaluates visitor routes across five major attractions at Universal's Islands of Adventure:
1. Hagrid's Magical Creatures Motorbike Adventure™
2. Jurassic World VelociCoaster
3. The Incredible Hulk Coaster®
4. The Amazing Adventures of Spider-Man®
5. Jurassic Park River Adventure™

---

## Algorithm Logic & Design

### 1. Asymmetric Time-Dependent Graph Construction
Unlike standard Euclidean routing networks, this model utilizes an asymmetric directed graph $G = (V, E)$. Walking durations ($w_{walk}(u, v)$) differ between the same nodes depending on direction due to theme park infrastructure, gate directionality, and pedestrian routing flows. Crowd dynamics are modeled using an empirical 63-timestamp wait-time dataset from Universal Orlando.

### 2. Evaluation Score Formulation
The classical A* formulation is modified to evaluate combined walking and real-time waiting costs:
$$f(v) = g(v) + h(v)$$

* **Actual Cost ($g(v)$)**: 
  $$g(v) = w_{walk}(current, v) + W_{queue}(v, t_{arr})$$
  where $t_{arr} = t_{current} + w_{walk}(current, v)$ is the exact arrival time at candidate $v$. This guarantees wait-time penalties are queried exactly at the simulated moment of physical arrival.
* **Admissible Heuristic ($h(v)$)**: 
  $$h(v, R') = \min_{i \in R'} w_{walk}(v, i), \quad h(v, \emptyset) = 0$$
  where $R'$ is the set of remaining unvisited attractions. The heuristic estimates the minimum possible walking time from the candidate attraction to the next closest unvisited attraction. By ignoring all future queue delays and only using the minimum spatial walking time, the heuristic **never overestimates** the true remaining sub-tour cost ($h(v) \leq \text{TrueCost}$), satisfying the condition of **admissibility**.

### 3. Sequential Greedy Routing Strategy
Since solving the full multi-destination TSP is NP-hard, the system employs a **Sequential Greedy A\*** approach to identify the locally optimal next attraction at each step. This achieves a highly efficient $O(|V|^2)$ search time per itinerary, rendering it viable for real-time mobile app recommendations.

### 4. Unguided Baseline Model
The A* Guided routes are benchmarked against an unguided **Random Walk Baseline** averaged across 20 independent trials (using seeds 0-19) to simulate realistic spontaneous visitor behaviors (*ride_enthusiasts* and *ride_favorers* who do not actively optimize itineraries).

---

## File Structure
* `main.py`: The core simulation script containing the asymmetric walking matrix, A* solver, Random Walk baseline generator, and result printouts.
* `data_antrean_5_wahana.csv`: Historical dataset containing wait times at 10-minute intervals for all five attractions.
* `hasil_simulasi.csv`: The numerical result table generated after executing the simulation.
* `comparison_chart.png`: Visual bar chart comparing A* Guided vs. Random Walk Baseline average times across 4 scenarios.

---

## Getting Started

### Prerequisites
Make sure you have Python 3 and the required libraries installed:
```bash
pip install pandas numpy matplotlib
```

### Running the Code
Run the main python script to execute the simulation, output the terminal comparison tables, and save the result documents:
```bash
python main.py
```
Upon running, it will automatically:
1. Load and parse the queue dataset.
2. Execute both A* Guided and Random Walk simulations across four operationally distinct scenarios (Early Morning, Peak Hours, Mid-Day, Afternoon).
3. Export the numerical summary to `hasil_simulasi.csv`.
4. Plot and save the comparison chart as `comparison_chart.png`.
