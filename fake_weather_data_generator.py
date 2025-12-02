import random
import pandas as pd

# -----------------------------------------
# Config
# -----------------------------------------
num_days = 256  # change this to however many rows you want
seasons = ["winter", "spring", "summer", "fall"]

# One base temperature table per season
temp_ranges = {
    "winter": (20.0, 45.0),
    "spring": (45.0, 70.0),
    "summer": (70.0, 100.0),
    "fall":   (45.0, 65.0),
}

# Humidity ranges
humidity_range_raining_yes = (95.0, 100.0)
humidity_range_raining_no  = (35.0, 100.0)

# -----------------------------------------
# Generate fake data
# -----------------------------------------
humidity_list = []
temperature_list = []
pressure_list = []
season_list = []
raining_list = []

for _ in range(num_days):
    season = random.choice(seasons)
    raining_str = random.choice(["True", "False"])
    is_raining = (raining_str == "True")

    base_low, base_high = temp_ranges[season]

    # Humidity
    if is_raining:
        hum_lo, hum_hi = humidity_range_raining_yes
    else:
        hum_lo, hum_hi = humidity_range_raining_no
    humidity = round(random.uniform(hum_lo, hum_hi), 2)

    # Temperature
    if is_raining:
        temp_lo = base_low * 0.9
        temp_hi = base_high * 0.95
    else:
        temp_lo = base_low
        temp_hi = base_high
    temperature = round(random.uniform(temp_lo, temp_hi), 2)

    # Pressure - lower when raining
    if is_raining:
        pressure = round(random.uniform(28.5 * 0.975, 31.5 * 0.95), 2)
    else:
        pressure = round(random.uniform(28.5, 31.5), 2)

    humidity_list.append(humidity)
    temperature_list.append(temperature)
    pressure_list.append(pressure)
    season_list.append(season)
    raining_list.append(raining_str)

df = pd.DataFrame({
    "humidity": humidity_list,
    "temperature": temperature_list,
    "pressure": pressure_list,
    "season": season_list,
    "raining": raining_list
})

# -----------------------------------------
# Sort and save
# -----------------------------------------
season_order = ["winter", "spring", "summer", "fall"]
raining_order = ["False", "True"]

df["season"] = pd.Categorical(df["season"], categories=season_order, ordered=True)
df["raining"] = pd.Categorical(df["raining"], categories=raining_order, ordered=True)

df = df.sort_values(by=["season", "raining"])

df.to_csv("weather_test.csv", index=False)
print(df.head())
