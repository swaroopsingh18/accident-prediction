import joblib
import pandas as pd
import numpy as np

# ── Weather stats from govdata.csv (India state-wise data 2014) ──────────────
WEATHER_STATS = {
    "Fine/Clear":  {"code": 0,  "avg_acc": 16763.3, "avg_killed": 4525.8,  "avg_injured": 16900.4, "mortality": 27.0},
    "Mist/Fog":    {"code": 1,  "avg_acc":   923.8, "avg_killed":  318.2,  "avg_injured":   787.0, "mortality": 34.4},
    "Cloudy":      {"code": 2,  "avg_acc":   852.4, "avg_killed":  253.4,  "avg_injured":   989.0, "mortality": 29.7},
    "Light Rain":  {"code": 3,  "avg_acc":  1448.6, "avg_killed":  409.6,  "avg_injured":  1461.6, "mortality": 28.3},
    "Heavy Rain":  {"code": 4,  "avg_acc":   978.8, "avg_killed":  314.3,  "avg_injured":  1007.8, "mortality": 32.1},
    "Flooding":    {"code": 5,  "avg_acc":   118.8, "avg_killed":   47.1,  "avg_injured":   114.4, "mortality": 39.6},
    "Hail/Sleet":  {"code": 6,  "avg_acc":    77.4, "avg_killed":   29.8,  "avg_injured":    74.5, "mortality": 38.5},
    "Snow":        {"code": 7,  "avg_acc":   169.0, "avg_killed":   44.5,  "avg_injured":   190.5, "mortality": 26.4},
    "Strong Wind": {"code": 8,  "avg_acc":   233.5, "avg_killed":   92.6,  "avg_injured":   233.9, "mortality": 39.7},
    "Dust Storm":  {"code": 9,  "avg_acc":   286.6, "avg_killed":   95.0,  "avg_injured":   287.2, "mortality": 33.2},
    "Very Hot":    {"code": 10, "avg_acc":   867.4, "avg_killed":  304.1,  "avg_injured":   986.3, "mortality": 35.1},
    "Very Cold":   {"code": 11, "avg_acc":   674.8, "avg_killed":  255.9,  "avg_injured":   675.7, "mortality": 37.9},
    "Others":      {"code": 12, "avg_acc":  2593.7, "avg_killed":  859.6,  "avg_injured":  2518.2, "mortality": 33.1},
}

MAX_ACC = 16763.3  # Fine/Clear baseline

def get_risk_percent(weather_name, model):
    stats = WEATHER_STATS[weather_name]
    input_df = pd.DataFrame([{
        "acc_2014":       stats["avg_acc"],
        "killed_2014":    stats["avg_killed"],
        "injured_2014":   stats["avg_injured"],
        "weather_code":   stats["code"],
        "mortality_rate": stats["mortality"] / 100.0,
    }])
    predicted_acc = model.predict(input_df)[0]
    risk = (predicted_acc / MAX_ACC) * 100
    risk = max(5.0, min(99.0, risk))
    return risk, predicted_acc

def print_banner():
    print("\n" + "=" * 55)
    print("      ROAD ACCIDENT PREDICTION SYSTEM")
    print("    Based on India State-wise Data (2014-2016)")
    print("=" * 55)

def choose_weather():
    print("\nSelect Current Weather Condition:\n")
    weather_list = list(WEATHER_STATS.keys())
    for i, w in enumerate(weather_list, 1):
        mort = WEATHER_STATS[w]["mortality"]
        print(f"  {i:2d}. {w:<15s}  (Mortality rate: {mort:.1f}%)")
    while True:
        choice = input("\nEnter number (1-13): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= 13:
            return weather_list[int(choice) - 1]
        print("  Invalid! Please enter a number between 1 and 13.")

def main():
    try:
        model = joblib.load("accident_model.pkl")
    except FileNotFoundError:
        print("\nERROR: accident_model.pkl not found!")
        print("Please run model.py first to train and save the model.")
        return

    print_banner()

    weather = choose_weather()
    stats   = WEATHER_STATS[weather]

    print(f"\nSelected: {weather}")
    print(f"  Historical avg accidents : {stats['avg_acc']:,.0f} per state")
    print(f"  Historical avg fatalities: {stats['avg_killed']:,.0f} per state")
    print(f"  Historical mortality rate: {stats['mortality']:.1f}%")

    risk_pct, pred_acc = get_risk_percent(weather, model)

    if risk_pct >= 60:
        level  = "HIGH RISK"
        advice = "Avoid travel if possible. Drive very slowly, use fog/hazard lights."
        bar    = "##########"
    elif risk_pct >= 30:
        level  = "MODERATE RISK"
        advice = "Drive carefully. Maintain safe distance. Reduce speed."
        bar    = "######...."
    else:
        level  = "LOW RISK"
        advice = "Relatively safe conditions. Always wear seatbelt."
        bar    = "###......."

    print("\n" + "=" * 55)
    print("           PREDICTION RESULT")
    print("=" * 55)
    print(f"  Weather Condition  : {weather}")
    print(f"  Predicted Accidents: ~{pred_acc:,.0f} (state-level estimate)")
    print(f"  Relative Risk      : {risk_pct:.1f}%  [{bar}]")
    print(f"  Risk Level         : {level}")
    print(f"\n  Advice: {advice}")
    print("=" * 55)

    print("\nWeather Risk Comparison (from your dataset):")
    print(f"  {'Weather':<15} {'Avg Accidents':>14} {'Mortality%':>11}")
    print(f"  {'-'*43}")
    sorted_weather = sorted(WEATHER_STATS.items(), key=lambda x: x[1]['avg_acc'], reverse=True)
    for w, s in sorted_weather:
        marker = " <<< YOU" if w == weather else ""
        print(f"  {w:<15} {s['avg_acc']:>14,.0f} {s['mortality']:>10.1f}%{marker}")
    print()

if __name__ == "__main__":
    main()
