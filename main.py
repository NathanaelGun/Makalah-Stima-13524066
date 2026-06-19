import pandas as pd
import numpy as np
import heapq
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# 1. KONSTANTA
# ─────────────────────────────────────────────

RIDES = [
    "Hagrid's Magical Creatures Motorbike Adventure™",
    "Jurassic World VelociCoaster",
    "The Incredible Hulk Coaster®",
    "The Amazing Adventures of Spider-Man®",
    "Jurassic Park River Adventure™"
]

# Singkatan untuk output
SHORT = {
    "Hagrid's Magical Creatures Motorbike Adventure™": "Hagrid's",
    "Jurassic World VelociCoaster":                     "VelociCoaster",
    "The Incredible Hulk Coaster®":                    "Hulk",
    "The Amazing Adventures of Spider-Man®":           "Spider-Man",
    "Jurassic Park River Adventure™":                   "Jurassic Park"
}

# Walk distance
WALK = np.array([
    [0,  4,  8,  9,  4],   # Hagrid's
    [4,  0, 11, 11,  4],   # VelociCoaster
    [8, 11,  0,  1,  8],   # Hulk
    [9, 11,  1,  0,  7],   # Spider-Man
    [4,  4,  8,  7,  0],   # Jurassic Park
], dtype=float)

# ─────────────────────────────────────────────
# 2. LOAD DAN PREPROCESS DATASET
# ─────────────────────────────────────────────

def load_dataset(filepath: str) -> dict:
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    index = {}
    for ride in RIDES:
        ride_df = df[df['ride_name'] == ride].sort_values('timestamp')
        index[ride] = dict(zip(ride_df['timestamp'], ride_df['wait_time']))
    
    print(f"Dataset loaded: {df['timestamp'].nunique()} timestamps")
    return index, sorted(df['timestamp'].unique())

def query_wait_time(index: dict, ride: str, t_arr: datetime) -> float:
    timestamps = list(index[ride].keys())
    if not timestamps:
        return 0.0
    
    # Cari timestamp terdekat (nearest neighbor lookup)
    nearest = min(timestamps, key=lambda ts: abs((ts - t_arr).total_seconds()))
    return float(index[ride][nearest])

# ─────────────────────────────────────────────
# 3. HEURISTIC FUNCTION h(n) UNTUK TSP ITINERARY
# ─────────────────────────────────────────────

def heuristic_tsp(current_idx: int, remaining_rides: list) -> float:
    """
    h(n) = Estimasi waktu jalan minimum ke wahana yang belum dikunjungi.
    
    Untuk TSP itinerary problem, tujuan akhir adalah "menyelesaikan semua wahana".
    Jadi h(n) adalah lower bound untuk waktu jalan ke salah satu wahana sisa.
    
    ADMISSIBLE: Karena pengunjung pasti perlu jalan + antri di wahana berikutnya,
    jarak jalan MINIMUM ke salah satu wahana adalah lower bound yang tidak overestimate.
    
    Returns:
        float: Jarak jalan minimum ke salah satu wahana sisa (dalam menit)
    """
    if not remaining_rides:
        return 0.0  # Kalau tidak ada wahana sisa, h(n) = 0
    
    # Cari jarak jalan terdekat dari current ke salah satu wahana yang belum dikunjungi
    min_walk = min(
        WALK[current_idx][RIDES.index(ride)] 
        for ride in remaining_rides
    )
    return min_walk

# ─────────────────────────────────────────────
# 4. DIRECT COST EVALUATION (1-HOP A*)
# ─────────────────────────────────────────────

def evaluate_next_ride(
    current: str,
    candidate: str,
    current_time: datetime,
    index: dict
) -> tuple[float, float, float, datetime]:
    """
    Evaluate direct cost (walk + wait) dari current ke candidate.
    Ini adalah A* recommendation untuk next ride.
    
    Returns:
        (total_cost, walk, wait, arrive_time)
    """
    current_idx = RIDES.index(current)
    candidate_idx = RIDES.index(candidate)
    
    w_walk = WALK[current_idx][candidate_idx]
    t_arrive = current_time + timedelta(minutes=w_walk)
    w_wait = query_wait_time(index, candidate, t_arrive)
    
    total_cost = w_walk + w_wait
    
    return total_cost, w_walk, w_wait, t_arrive

# ─────────────────────────────────────────────
# 5. SIMULASI ITINERARY LENGKAP (A* GUIDED)
# ─────────────────────────────────────────────

def estimate_remaining_walk(current: str, candidate: str, remaining: list) -> float:
    """
    Estimasi kasar total walking jika kita pergi ke candidate sekarang,
    lalu nearest-neighbor untuk sisa wahana.
    Ini membuat A* mempertimbangkan konsekuensi posisi geografis.
    """
    if not remaining:
        return 0.0
    
    # Walking ke candidate
    c_idx = RIDES.index(current)
    cand_idx = RIDES.index(candidate)
    walk = WALK[c_idx][cand_idx]
    
    # Estimasi greedy sisa dari candidate
    pos = candidate
    others = [r for r in remaining if r != candidate]
    total_remaining = 0.0
    while others:
        pos_idx = RIDES.index(pos)
        nearest = min(others, key=lambda r: WALK[pos_idx][RIDES.index(r)])
        total_remaining += WALK[pos_idx][RIDES.index(nearest)]
        others.remove(nearest)
        pos = nearest
    
    return walk + total_remaining


def simulate_astar_itinerary(start, t_start, index):
    visited = [start]
    remaining = [r for r in RIDES if r != start]
    
    current = start
    current_time = t_start
    total_wait = 0.0
    total_walk = 0.0
    log = []

    first_wait = query_wait_time(index, start, t_start)
    total_wait += first_wait
    current_time += timedelta(minutes=first_wait)
    log.append({'step': 0, 'ride': start, 'arrive': t_start,
                'wait': first_wait, 'walk_from_prev': 0})

    step = 1
    while remaining:
        current_idx = RIDES.index(current)
        best_cost   = float('inf')
        best_target = None
        best_walk   = 0
        best_wait   = 0
        best_arrive = None

        for candidate in remaining:
            cand_idx = RIDES.index(candidate)
            w_walk   = WALK[current_idx][cand_idx]
            t_arr    = current_time + timedelta(minutes=w_walk)
            wait     = query_wait_time(index, candidate, t_arr)

            # g(n) = walk ke candidate + wait di candidate
            # h(n) = estimasi walking sisa itinerary dari candidate
            g = w_walk + wait
            h = estimate_remaining_walk(current, candidate, 
                    [r for r in remaining if r != candidate])
            f = g + h

            if f < best_cost:
                best_cost   = f
                best_target = candidate
                best_walk   = w_walk
                best_wait   = wait
                best_arrive = t_arr

        total_walk   += best_walk
        total_wait   += best_wait
        current_time  = best_arrive + timedelta(minutes=best_wait)

        log.append({
            'step': step, 'ride': best_target,
            'arrive': best_arrive, 'wait': best_wait,
            'walk_from_prev': best_walk
        })

        visited.append(best_target)
        remaining.remove(best_target)
        current = best_target
        step += 1

    return {
        'strategy': 'A* Guided',
        'total_walk': total_walk,
        'total_wait': total_wait,
        'total_time': total_walk + total_wait,
        'order': visited,
        'log': log
    }

# ─────────────────────────────────────────────
# 6. SIMULASI BASELINE (UNGUIDED - RANDOM ORDER)
# ─────────────────────────────────────────────

import random

def simulate_random_baseline(
    start: str,
    t_start: datetime,
    index: dict,
    seed: int = 42
) -> dict:
    """
    Baseline: pengunjung mengunjungi wahana dalam urutan acak.
    Ini lebih realistis menggambarkan perilaku tanpa panduan —
    orang tidak selalu pergi ke wahana terdekat, mereka ikut kerumunan
    atau pergi berdasarkan preferensi acak.
    
    Args:
        start: Wahana awal
        t_start: Waktu mulai
        index: Dataset index
        seed: Random seed untuk reproducibility
    
    Returns:
        dict: Hasil simulasi dengan detail step-by-step
    """
    random.seed(seed)
    remaining = [r for r in RIDES if r != start]
    random.shuffle(remaining)
    order = [start] + remaining
    
    current = start
    current_time = t_start
    total_walk = 0.0
    total_wait = 0.0
    log = []
    
    first_wait = query_wait_time(index, start, t_start)
    total_wait += first_wait
    current_time += timedelta(minutes=first_wait)
    log.append({
        'step': 0, 'ride': start,
        'arrive': t_start, 'wait': first_wait,
        'walk_from_prev': 0
    })
    
    for step, (from_ride, to_ride) in enumerate(zip(order, order[1:]), 1):
        from_idx = RIDES.index(from_ride)
        to_idx = RIDES.index(to_ride)
        walk = WALK[from_idx][to_idx]
        arrive = current_time + timedelta(minutes=walk)
        wait = query_wait_time(index, to_ride, arrive)
        
        total_walk += walk
        total_wait += wait
        current_time = arrive + timedelta(minutes=wait)
        
        log.append({
            'step': step, 'ride': to_ride,
            'arrive': arrive, 'wait': wait,
            'walk_from_prev': walk
        })
    
    return {
        'strategy': 'Unguided (Random)',
        'total_walk': total_walk,
        'total_wait': total_wait,
        'total_time': total_walk + total_wait,
        'order': order,
        'log': log
    }

def simulate_random_baseline_avg(
    start: str,
    t_start: datetime,
    index: dict,
    n_trials: int = 20
) -> dict:
    """
    Jalankan baseline random berkali-kali dengan seed berbeda,
    ambil rata-rata, lalu return satu run representatif.
    
    Args:
        start: Wahana awal
        t_start: Waktu mulai
        index: Dataset index
        n_trials: Jumlah percobaan acak (default 20)
    
    Returns:
        dict: Hasil run yang paling dekat dengan rata-rata, plus statistik
    """
    results = []
    for seed in range(n_trials):
        r = simulate_random_baseline(start, t_start, index, seed=seed)
        results.append(r['total_time'])
    
    avg_time = sum(results) / len(results)
    worst_time = max(results)
    best_time = min(results)
    range_time = worst_time - best_time
    
    # Calculate variance metrics
    variance = sum((x - avg_time) ** 2 for x in results) / len(results)
    stdev = variance ** 0.5
    cv = stdev / avg_time * 100  # Coefficient of variation
    
    # Ambil satu run representatif (yang paling dekat ke rata-rata)
    rep_seed = min(
        range(n_trials),
        key=lambda s: abs(results[s] - avg_time)
    )
    rep = simulate_random_baseline(start, t_start, index, seed=rep_seed)
    
    # Tambahkan statistik ke hasil
    rep['strategy'] = f'Unguided (Random, avg of {n_trials} trials)'
    rep['total_time_avg'] = avg_time
    rep['total_time_best'] = best_time
    rep['total_time_worst'] = worst_time
    rep['total_time_range'] = range_time
    rep['total_time_stdev'] = stdev
    rep['total_time_cv'] = cv
    rep['rep_seed'] = rep_seed
    
    return rep

def simulate_unguided_itinerary(
    start: str,
    t_start: datetime,
    index: dict
) -> dict:
    """
    Wrapper yang menjalankan random baseline dengan averaging.
    Untuk kompatibilitas dengan fungsi run_all_scenarios().
    """
    return simulate_random_baseline_avg(start, t_start, index, n_trials=20)

# ─────────────────────────────────────────────
# 7. EVALUASI DAN OUTPUT
# ─────────────────────────────────────────────

def print_result(result: dict):
    print(f"\n{'='*55}")
    print(f"  {result['strategy']}")
    print(f"{'='*55}")
    print(f"  {'Step':<5} {'Wahana':<30} {'Walk':>5} {'Wait':>5}")
    print(f"  {'-'*48}")
    for entry in result['log']:
        ride_short = SHORT.get(entry['ride'], entry['ride'])
        print(f"  {entry['step']:<5} {ride_short:<30} "
              f"{entry['walk_from_prev']:>4.0f}m {entry['wait']:>4.0f}m")
    print(f"  {'-'*48}")
    print(f"  Total jalan      : {result['total_walk']:.0f} menit")
    print(f"  Total nunggu     : {result['total_wait']:.0f} menit")
    print(f"  TOTAL KESELURUHAN: {result['total_time']:.0f} menit")
    
    # Tampilkan statistik jika ada (untuk baseline random)
    if 'total_time_avg' in result:
        print(f"\n  [Statistik Random Baseline - 20 trials]")
        print(f"  Rata-rata      : {result['total_time_avg']:.0f} menit")
        print(f"  Stdev          : {result['total_time_stdev']:.1f} menit")
        print(f"  Coeff Variation: {result['total_time_cv']:.1f}%")
        print(f"  Range          : {result['total_time_best']:.0f} - {result['total_time_worst']:.0f} menit (span: {result['total_time_range']:.0f}m)")
        print(f"  Run representatif: seed={result['rep_seed']} (diff: {abs(result['total_time'] - result['total_time_avg']):.1f}m dari avg)")

def compare_results(r_astar: dict, r_baseline: dict):
    # Gunakan rata-rata baseline jika tersedia, otherwise gunakan total_time
    baseline_time = r_baseline.get('total_time_avg', r_baseline['total_time'])
    delta = baseline_time - r_astar['total_time']
    pct   = delta / baseline_time * 100
    print(f"\n{'='*55}")
    print(f"  Perbandingan A* vs Baseline (Random)")
    print(f"{'='*55}")
    print(f"  A* total           : {r_astar['total_time']:.0f} menit")
    print(f"  Baseline total     : {r_baseline['total_time']:.0f} menit")
    if 'total_time_avg' in r_baseline:
        print(f"  Baseline rata-rata : {baseline_time:.0f} menit (dari 20 trials)")
    print(f"  Penghematan        : {delta:.0f} menit ({pct:.1f}%)")

# ─────────────────────────────────────────────
# 8. MAIN — JALANKAN SEMUA SKENARIO
# ─────────────────────────────────────────────

def run_all_scenarios(index: dict, timestamps: list) -> list:
    """
    Menjalankan 4 skenario berbeda dengan START dan T_START yang berbeda-beda.
    Return: list of comparison results untuk tabel
    """
    scenarios = [
        ("The Incredible Hulk Coaster®", 0, "08:00 (Early morning)"),
        ("Hagrid's Magical Creatures Motorbike Adventure™", 15, "10:00 (Peak hours)"),
        ("The Amazing Adventures of Spider-Man®", 25, "12:00 (Mid-day)"),
        ("Jurassic World VelociCoaster", 40, "14:00 (Afternoon)")
    ]
    
    results_table = []
    
    for i, (start_ride, ts_idx, time_label) in enumerate(scenarios, 1):
        if ts_idx >= len(timestamps):
            ts_idx = len(timestamps) - 1
        
        t_start = timestamps[ts_idx]
        
        print(f"\n\n{'='*70}")
        print(f"  SKENARIO {i}: Mulai dari {SHORT[start_ride]} — {time_label}")
        print(f"{'='*70}")
        print(f"  Waktu mulai: {t_start}")
        
        r_astar    = simulate_astar_itinerary(start_ride, t_start, index)
        r_baseline = simulate_unguided_itinerary(start_ride, t_start, index)
        
        print_result(r_astar)
        print_result(r_baseline)
        
        # Gunakan rata-rata baseline untuk perbandingan
        baseline_time = r_baseline.get('total_time_avg', r_baseline['total_time'])
        delta = baseline_time - r_astar['total_time']
        pct   = delta / baseline_time * 100
        
        print(f"\n  Perbandingan A* vs Baseline (Random)")
        print(f"  A* total         : {r_astar['total_time']:.0f} menit")
        if 'total_time_avg' in r_baseline:
            print(f"  Baseline total   : {r_baseline['total_time']:.0f} menit")
            print(f"  Baseline rata-rata: {baseline_time:.0f} menit (avg dari 20 trials)")
        else:
            print(f"  Baseline total   : {baseline_time:.0f} menit")
        print(f"  Penghematan      : {delta:.0f} menit ({pct:.1f}%)")
        
        results_table.append({
            'no': i,
            'start': SHORT[start_ride],
            'time': time_label,
            'astar': r_astar['total_time'],
            'baseline': r_baseline['total_time'],
            'baseline_avg': baseline_time,
            'baseline_range': f"{r_baseline.get('total_time_best', r_baseline['total_time']):.0f}-{r_baseline.get('total_time_worst', r_baseline['total_time']):.0f}",
            'baseline_cv': r_baseline.get('total_time_cv', 0),
            'savings': delta,
            'pct': pct
        })
    
    return results_table

def print_results_table(results: list):
    """
    Menampilkan hasil dalam bentuk tabel untuk makalah.
    Membandingkan A* dengan baseline rata-rata (jika ada statistik dari random trials).
    """
    print(f"\n\n{'='*160}")
    print("  TABEL HASIL SIMULASI — PERBANDINGAN A* GUIDED VS UNGUIDED RANDOM BASELINE")
    print(f"{'='*160}")
    print(f"  {'No':<4} {'Start':<20} {'Waktu':<20} {'A* (min)':<12} {'Baseline (min)':<15} {'Baseline Avg':<15} {'Range':<15} {'CV%':<8} {'Selisih':<12} {'Efisiensi':<12}")
    print(f"  {'-'*160}")
    
    positive_count = 0
    for r in results:
        baseline_avg_str = f"{r.get('baseline_avg', r['baseline']):.0f}" if 'baseline_avg' in r else "-"
        baseline_range_str = r.get('baseline_range', '-')
        baseline_cv_str = f"{r.get('baseline_cv', 0):.1f}" if 'baseline_cv' in r else "-"
        efficiency = "* LEBIH BAIK" if r['pct'] > 0 else ""
        print(f"  {r['no']:<4} {r['start']:<20} {r['time']:<20} "
              f"{r['astar']:<12.0f} {r['baseline']:<15.0f} {baseline_avg_str:<15} {baseline_range_str:<15} {baseline_cv_str:<8} {r['savings']:<12.0f} {r['pct']:>6.1f}% {efficiency:<12}")
        if r['pct'] > 0:
            positive_count += 1
    
    print(f"  {'-'*160}")
    
    # Hitung statistik
    avg_pct = sum(r['pct'] for r in results) / len(results)
    total_savings = sum(r['savings'] for r in results)
    
    print(f"  Rata-rata efisiensi: {avg_pct:.1f}%  |  Total penghematan: {total_savings:.0f} menit  |  Skenario A* lebih baik: {positive_count}/{len(results)}")
    print(f"  [Baseline = Random Walk; Avg = Rata-rata dari 20 random trials; CV = Coefficient of Variation]")
    print(f"  [High CV menunjukkan random baseline sangat tidak stabil; A* deterministic (CV=0)]")
    print(f"{'='*160}\n")

def save_results_to_csv(results: list, filename: str = "hasil_simulasi.csv"):
    """
    Menyimpan hasil simulasi ke file CSV untuk dilampirkan di makalah.
    """
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['No', 'Start Ride', 'Time', 'A* (min)', 'Baseline (min)', 'Savings (min)', 'Efficiency (%)'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'No': r['no'],
                'Start Ride': r['start'],
                'Time': r['time'],
                'A* (min)': f"{r['astar']:.0f}",
                'Baseline (min)': f"{r['baseline']:.0f}",
                'Savings (min)': f"{r['savings']:.0f}",
                'Efficiency (%)': f"{r['pct']:.1f}%"
            })
    print(f"[OK] Hasil disimpan ke: {filename}")

def generate_comparison_chart(results: list, filename: str = "comparison_chart.png"):
    """
    Men-generate chart perbandingan A* Guided vs Baseline dengan Matplotlib.
    """
    import matplotlib.pyplot as plt
    
    scenarios = [f"{r['start']}\n({r['time'].split(' ')[0]})" for r in results]
    astar = [r['astar'] for r in results]
    random_avg = [r['baseline_avg'] for r in results]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, astar, width, label='A* Guided', color='#2196F3')
    ax.bar(x + width/2, random_avg, width, 
           label='Random Baseline (avg 20 trials)', color='#FF9800', alpha=0.8)
    
    ax.set_ylabel('Total Time (minutes)')
    ax.set_title('A* Guided vs. Unguided Random Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (a, r) in enumerate(zip(astar, random_avg)):
        delta = r - a
        color = 'green' if delta > 0 else 'red'
        sign = '+' if delta > 0 else ''
        ax.annotate(f'{sign}{delta:.0f}min', xy=(i, max(a,r)+5), 
                    ha='center', fontsize=9, color=color, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafik perbandingan disimpan ke: {filename}")

if __name__ == "__main__":
    CSV_PATH = "data_antrean_5_wahana.csv"
    index, timestamps = load_dataset(CSV_PATH)
    
    results_table = run_all_scenarios(index, timestamps)
    print_results_table(results_table)
    save_results_to_csv(results_table)
    generate_comparison_chart(results_table)