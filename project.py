import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

#   ENGINE PARAMETERS
V_max = 1.0e-3  # Max volume [m^3]
compression_ratio = 18.0
V_min = V_max / compression_ratio
engine_speed = 1800  # RPM
cycle_duration = 60.0 / engine_speed  # [s]
A = 0.01  # Piston area [m^2]

#   COMBUSTION SETTINGS
mechanism = 'gri30.yaml'
composition = 'CH4:1, O2:2, N2:7.52'
P_initial = 1e5  # [Pa]
T_initial_values = [1500.0, 1800.0, 2500.0]  # [K]

#   VOLUME FUNCTIONS
def cylinder_volume(t, t_comp, t_burn, t_expand):
    if t < t_comp:
        return V_max - 0.5 * (V_max - V_min) * (1 - np.cos(np.pi * t / t_comp))
    elif t < t_comp + t_burn:
        return V_min
    else:
        t_rel = t - (t_comp + t_burn)
        return V_min + 0.5 * (V_max - V_min) * (1 - np.cos(np.pi * t_rel / t_expand))

def volume_derivative(t, t_comp, t_burn, t_expand):
    if t < t_comp:
        return 0.5 * np.pi * (V_max - V_min) / t_comp * np.sin(np.pi * t / t_comp)
    elif t < t_comp + t_burn:
        return 0.0
    else:
        t_rel = t - (t_comp + t_burn)
        return 0.5 * np.pi * (V_max - V_min) / t_expand * np.sin(np.pi * t_rel / t_expand)

#   TIME SETTINGS
t_comp = 0.01      # 10 ms compression
t_burn = 0.002     # 2 ms combustion
t_expand = 0.015   # 15 ms expansion
freeze_time = t_comp + t_burn + t_expand
end_time = freeze_time + 0.005
dt = 1e-5  # Time step [s]

#   STORAGE FOR RESULTS
results = []

for T_initial in T_initial_values:
    gas = ct.Solution(mechanism)
    gas.TPX = T_initial, P_initial, composition

    reactor = ct.IdealGasReactor(gas, volume=V_max, energy='on')
    env = ct.Reservoir(ct.Solution(mechanism))
    wall = ct.Wall(reactor, env)
    wall.area = A
    wall.velocity = 0.0

    sim = ct.ReactorNet([reactor])

    times, T, P, NO, NO2, NO_ppm, NO2_ppm = [], [], [], [], [], [], []
    t = 0.0

    while t < end_time:
        wall.velocity = volume_derivative(t, t_comp, t_burn, t_expand) / A

        if t > freeze_time:
            reactor.chemistry_enabled = False

        sim.advance(t)

        times.append(t)
        T.append(reactor.T)
        P.append(reactor.thermo.P)
        no_x = reactor.thermo['NO'].X[0]
        no2_x = reactor.thermo['NO2'].X[0]
        NO.append(no_x)
        NO2.append(no2_x)
        NO_ppm.append(no_x * 1e6)
        NO2_ppm.append(no2_x * 1e6)
        t += dt

    results.append({
        'T_init': T_initial,
        'time': times,
        'T': T,
        'P': P,
        'NO': NO,
        'NO2': NO2,
        'NO_ppm': NO_ppm,
        'NO2_ppm': NO2_ppm
    })


#   PLOTTING
plt.figure(figsize=(10, 6))
for res in results:
    plt.plot(res['time'], res['NO_ppm'], label=f"T₀ = {res['T_init']} K")
plt.xlabel('Time [s]')
plt.ylabel('NO [ppm]')
plt.title('NO (in ppm) vs Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for res in results:
    plt.plot(res['time'], res['NO2_ppm'], label=f"T₀ = {res['T_init']} K")
plt.xlabel('Time [s]')
plt.ylabel('NO2 [ppm]')
plt.title('NO2 (in ppm) vs Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

temps = [res['T_init'] for res in results]
max_nox = [max(np.array(res['NO_ppm']) + np.array(res['NO2_ppm'])) for res in results]

plt.figure(figsize=(8, 5))
plt.plot(temps, max_nox, 'o-', color='darkred')
plt.xlabel('Initial Temperature [K]')
plt.ylabel('Max NOx [ppm]')
plt.title('Maximum NOx vs Initial Temperature')
plt.grid()
plt.tight_layout()
plt.show()

print("Maximum NO and NO2 concentrations (in ppm) for each initial temperature:")
print("{:<15} {:<15} {:<15}".format("T_init [K]", "NO [ppm]", "NO2 [ppm]"))
for res in results:
    T_init = res['T_init']
    max_NO = max(res['NO_ppm'])
    max_NO2 = max(res['NO2_ppm'])
    print("{:<15} {:<15.2f} {:<15.2f}".format(T_init, max_NO, max_NO2))
