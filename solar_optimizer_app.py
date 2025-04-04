import subprocess
import sys
import importlib

def install_and_import(package, import_as=None):
    try:
        importlib.import_module(import_as or package)
    except ImportError:
        print(f"Installing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[import_as or package] = importlib.import_module(import_as or package)

# --- Install necessary packages ---
install_and_import('matplotlib')
install_and_import('numpy')
install_and_import('pandas')
install_and_import('streamlit')
install_and_import('pymoo')
install_and_import('scikit-learn', 'sklearn')

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import io

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

# Load datasets
panel_df = pd.read_csv("solar_panels.csv")
inverter_df = pd.read_csv("inverters.csv")
battery_df = pd.read_csv("batteries.csv")
load_df = pd.read_csv("load_profile.csv")
irradiance_df = pd.read_csv("irradiance_data.csv")

# Sidebar Inputs
st.sidebar.header("🔧 System Design Inputs")
user_load = st.sidebar.slider("Average Daily Load (kWh)", 10.0, 50.0, float(load_df["Load_kWh"].sum()))
user_irradiance = st.sidebar.slider("Average Irradiance (kWh/m²)", 3.0, 7.5, float(irradiance_df["Irradiance_kWh/m2"].mean()))
user_roof_area = st.sidebar.number_input("Available Roof Area (m²)", min_value=10.0, value=40.0)
user_budget = st.sidebar.number_input("Max System Budget ($)", min_value=1000, value=10000)
battery_backup = st.sidebar.slider("Minimum Battery Backup (kWh)", 0.0, 20.0, 5.0)

# Optional brand filtering
st.sidebar.markdown("#### ⚙️ Component Preferences")
panel_brands = panel_df["Brand"].unique().tolist()
inverter_brands = inverter_df["Brand"].unique().tolist()
battery_brands = battery_df["Brand"].unique().tolist()

selected_panel_brand = st.sidebar.selectbox("Preferred Panel Brand", ["Any"] + panel_brands)
selected_inverter_brand = st.sidebar.selectbox("Preferred Inverter Brand", ["Any"] + inverter_brands)
selected_battery_brand = st.sidebar.selectbox("Preferred Battery Brand", ["Any"] + battery_brands)

if selected_panel_brand != "Any":
    panel_df = panel_df[panel_df["Brand"] == selected_panel_brand].reset_index(drop=True)
if selected_inverter_brand != "Any":
    inverter_df = inverter_df[inverter_df["Brand"] == selected_inverter_brand].reset_index(drop=True)
if selected_battery_brand != "Any":
    battery_df = battery_df[battery_df["Brand"] == selected_battery_brand].reset_index(drop=True)

st.title("☀️ AI-Optimized Solar PV System Design")

# Optimization problem
class PVSystemDesignProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=5,
            n_obj=2,
            n_constr=1,
            xl=np.array([0, 0, 0, 5, 1]),
            xu=np.array([len(panel_df)-1, len(inverter_df)-1, len(battery_df)-1, 20, 3])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        panel = panel_df.iloc[int(x[0])]
        inverter = inverter_df.iloc[int(x[1])]
        battery = battery_df.iloc[int(x[2])]
        num_panels = int(x[3])
        num_batteries = int(x[4])

        gen_kwh = num_panels * panel["Power_W"] / 1000 * user_irradiance
        usable_storage = num_batteries * battery["Capacity_kWh"] * battery["DoD"] * battery["Efficiency"]
        energy_deficit = max(0, user_load - gen_kwh - usable_storage)

        total_cost = (
            num_panels * panel["Cost_USD"] +
            num_batteries * battery["Cost_USD"] +
            inverter["Cost_USD"]
        )

        inverter_ok = inverter["Rated_Power_kW"] * 1000 - (num_panels * panel["Power_W"])
        roof_ok = (num_panels * panel["Area_m2"]) <= user_roof_area
        cost_ok = total_cost <= user_budget
        backup_ok = usable_storage >= battery_backup

        violations = 0
        if inverter_ok < 0 or not roof_ok or not cost_ok or not backup_ok:
            violations += 1

        out["F"] = [total_cost, energy_deficit]
        out["G"] = [violations]

# Run optimization
with st.spinner("⚙️ Running optimization..."):
    problem = PVSystemDesignProblem()
    algorithm = NSGA2(
        pop_size=50,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=3.0),
        eliminate_duplicates=True
    )
    res = minimize(problem, algorithm, get_termination("n_gen", 30), seed=1, verbose=False)

# Display results
st.success("✅ Optimization complete!")

st.subheader("🏆 Top Optimized Configurations")
top_data = []

for i, sol in enumerate(res.X[:5]):
    panel = panel_df.iloc[int(sol[0])]
    inverter = inverter_df.iloc[int(sol[1])]
    battery = battery_df.iloc[int(sol[2])]
    num_panels = int(sol[3])
    num_batteries = int(sol[4])
    cost, deficit = res.F[i]

    top_data.append({
        "Panel_Model": panel["Model"],
        "Inverter_Model": inverter["Model"],
        "Battery_Model": battery["Model"],
        "Panels": num_panels,
        "Batteries": num_batteries,
        "Total_Cost": cost,
        "Energy_Deficit_kWh": deficit
    })

    with st.expander(f"🔹 Solution {i+1}: ${cost:.2f} | Deficit: {deficit:.2f} kWh"):
        st.write(f"**Panel:** {panel['Model']}")
        st.write(f"**Inverter:** {inverter['Model']}")
        st.write(f"**Battery:** {battery['Model']}")
        st.write(f"**Panels:** {num_panels}")
        st.write(f"**Batteries:** {num_batteries}")
        st.write(f"**Total Cost:** ${cost:.2f}")
        st.write(f"**Energy Deficit:** {deficit:.2f} kWh")

# Energy flow chart
best_sol = res.X[0]
panel = panel_df.iloc[int(best_sol[0])]
battery = battery_df.iloc[int(best_sol[2])]
num_panels = int(best_sol[3])
num_batteries = int(best_sol[4])
gen_kwh = num_panels * panel["Power_W"] / 1000 * user_irradiance
storage_kwh = num_batteries * battery["Capacity_kWh"] * battery["DoD"] * battery["Efficiency"]

st.subheader("📊 Energy Flow of Best Configuration")
fig, ax = plt.subplots()
ax.bar(["Generation", "Battery Storage", "Daily Load"], [gen_kwh, storage_kwh, user_load], color=["orange", "green", "blue"])
ax.set_ylabel("kWh")
st.pyplot(fig)

# CO2 savings
co2_per_kwh = 0.7
co2_savings = gen_kwh * co2_per_kwh
st.metric("🌱 CO₂ Savings (kg/day)", f"{co2_savings:.2f}")

# Export top results
st.subheader("💾 Export Top 5 Solutions")
csv_buffer = io.StringIO()
pd.DataFrame(top_data).to_csv(csv_buffer, index=False)
st.download_button("Download CSV", data=csv_buffer.getvalue(), file_name="top_solar_configurations.csv", mime="text/csv")
