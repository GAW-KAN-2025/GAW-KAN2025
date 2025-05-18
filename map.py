import folium
import pandas as pd

# 假设lons, lats已如上读取
# 以PEMS-BAY为例
csv_path = './data/PEMS-BAY/graph_sensor_locations_bay.csv'
loc_df = pd.read_csv(csv_path, header=None)
lats = loc_df[1].values
lons = loc_df[2].values

# 以第一个点为中心
m = folium.Map(location=[lats[0], lons[0]], zoom_start=12, tiles='CartoDB positron')
for lat, lon in zip(lats, lons):
    folium.CircleMarker([lat, lon], radius=3, color='red').add_to(m)

m.save('map_with_nodes.html')