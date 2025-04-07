import pandas as pd
import random
from datetime import datetime, timedelta

def generate_sensor_data(num_rows=100):
    data = []
    base_time = datetime.now()

    for i in range(num_rows):
        source_id = random.randint(1, 5)
        source_address = f"192.168.1.{source_id}"
        source_type = random.choice(['anemometer', 'wind_vane'])
        source_location = random.choice(['loc1', 'loc2', 'loc3'])
        operation = 'send'
        timestamp = base_time + timedelta(seconds=i * 60)
        wind_speed = random.uniform(5, 50)
        wind_direction = random.choice(['N', 'S', 'E', 'W'])

        # Randomly decide if this row is an attack (balanced 50/50)
        attack = random.choice(['Yes', 'No'])

        data.append([
            source_id, source_address, source_type,
            source_location, operation, timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            round(wind_speed, 2), wind_direction, attack
        ])

    columns = ['source_id', 'source_address', 'source_type', 'source_location',
               'operation', 'timestamp', 'wind_speed', 'wind_direction', 'attack']

    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == "__main__":
    df = generate_sensor_data(100)  # generate 100 rows
    df.to_csv('dataset/sensor_data.csv', index=False)
    print("✅ Fake sensor data saved to dataset/sensor_data.csv")
