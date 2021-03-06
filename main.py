import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import folium
import re
from collections import defaultdict


def prepare_header(data):
    with open(data, 'w') as wf:
        wf.write('ID,NAME,YEAR,MONTH,LAT,LONG,MAX WIND,MIN PRESSURE,CATEGORY\n')


def data_extraction(data, mode='atlantic'):
    ORIGINAL_DATA = ''
    if mode == 'atlantic':
        ORIGINAL_DATA = 'data/atlantic.csv'
    elif mode == 'pacific':
        ORIGINAL_DATA = 'data/pacific.csv'

    unnamed_counter = 0
    hurricane_name = ''
    hurricane_id = -1
    with open(ORIGINAL_DATA, 'r') as f:
        for line in f:
            line = line.rstrip().split(',')    
            year = 0
            month = 0
            lat = 0
            long = 0
            max_wind = 0
            min_pressure = 0
            cat = 0
            # header line has length = 4
            if len(line) == 4:
                if line[1].replace(' ', '') == 'UNNAMED':
                    hurricane_name = 'UNNAMED_' + str(unnamed_counter)
                else:
                    hurricane_name = line[1].replace(' ', '')
                unnamed_counter += 1
                hurricane_id += 1
            else:
                year = line[0][:4].replace(' ', '')
                month = line[0][4:6].replace(' ', '')
                lat = line[4].replace(' ', '')[:-1]
                long = '-' + line[5].replace(' ', '')[:-1]
                max_wind = int(line[6].replace(' ', ''))
                if 74 <= max_wind <= 95:
                    cat = 1
                elif 96 <= max_wind <= 110:
                    cat = 2
                elif 111 <= max_wind <= 130:
                    cat = 3
                elif 131 <= max_wind <= 155:
                    cat = 4
                elif max_wind >= 155:
                    cat = 5
                else:
                    cat = 0
                min_pressure = int(line[7].replace(' ', ''))
                if min_pressure == -999:
                    min_pressure = 0
                with open(data, 'a') as wf:
                    wf.write(str(hurricane_id) + ',' + hurricane_name + ',' +  year + ',' + month +
                            ',' + lat + ',' + long + ',' + str(max_wind) + ',' + str(min_pressure) +
                            ',' + str(cat) + '\n')


def get_CO2(data):
    df = pd.read_csv(data)
    df = df.drop('unc', axis=1)
    return df


def graph_hurricane_category(hurricane_df, mode='atlantic'):
    grouped = hurricane_df.groupby('CATEGORY')['ID'].nunique()
    grouped.plot(kind='bar')
    ax = plot.gca()
    plot.xticks(rotation=0)
    plot.ylabel('Counts')
    plot.xlabel('Hurricane Category')
    title = ''
    if mode == 'atlantic':
        title = 'Frequency of Hurricane Categories in the Atlantic Ocean'
    elif mode == 'pacific':
        title = 'Frequency of Hurricane Categories in the Northeast and North Central Pacific Oceans'
    plot.title(title)
    textstr = '\n'.join(('Category 0: Winds less than 74 mph',
                        'Category 1: Winds 74 to 95 mph',
                        'Category 2: Winds 96 to 110 mph',
                        'Category 3: Winds 111 to 130 mph',
                        'Category 4: Winds 131 to 155 mph',
                        'Category 5: Winds greater than 155 mph'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1.25), ha='center')
    plot.show()


def graph_hurricane_month(hurricane_df, mode='atlantic'):
    grouped = hurricane_df.groupby('MONTH')['ID'].nunique()
    grouped.plot(kind='bar')
    title = ''
    if mode == 'atlantic':
        title = 'Monthly Frequency of Hurricanes in the Atlantic Ocean'
    elif mode == 'pacific':
        title = 'Monthly Frequency of Hurricanes in the Northeast and North Central Pacific Oceans'
    plot.xticks(rotation=0)
    plot.ylabel('Counts')
    plot.xlabel('Month')
    plot.title(title)
    plot.show()


def storm_track(hurricane_df, mode='atlantic'):
    temp_df = hurricane_df[['ID', 'NAME', 'YEAR', 'CATEGORY']]
    cat_0 = temp_df.loc[hurricane_df['CATEGORY'] == 0].sample(n=3)
    cat_1 = temp_df.loc[hurricane_df['CATEGORY'] == 1].sample(n=3)
    cat_2 = temp_df.loc[hurricane_df['CATEGORY'] == 2].sample(n=3)
    cat_3 = temp_df.loc[hurricane_df['CATEGORY'] == 3].sample(n=3)
    cat_4 = temp_df.loc[hurricane_df['CATEGORY'] == 4].sample(n=3)
    cat_5 = temp_df.loc[hurricane_df['CATEGORY'] == 5]
    hurricane_coor = cat_0.values.tolist()
    hurricane_coor.extend(cat_1.values.tolist())
    hurricane_coor.extend(cat_2.values.tolist())
    hurricane_coor.extend(cat_3.values.tolist())
    hurricane_coor.extend(cat_4.values.tolist())
    hurricane_coor.extend(cat_5.values.tolist())
    hurricane_list = []
    for i in hurricane_coor:
        id, name, year, category = i
        coor = hurricane_df.loc[hurricane_df['ID'] == id][['LAT', 'LONG']].values.tolist()
        hurricane_list.append((coor, name, year, category))

    map = folium.Map(zoom_start=6)
    for coord in hurricane_list:
        coordinate, name, year, category = coord
        color = ''
        if category == 0:
            color = 'gray'
        elif category == 1:
            color = 'blue'
        elif category == 2:
            color = 'orange'
        elif category == 3:
            color = 'lightred'
        elif category == 4:
            color = 'red'
        elif category == 5:
            color = 'darkred'
        tooltip = name + ' in ' + str(year) + ' category ' + str(category)
        my_PolyLine=folium.PolyLine(locations=coordinate, weight=3, color=color, tooltip=tooltip)
        map.add_child(my_PolyLine)
        folium.Marker(location=coordinate[0], tooltip=name + ' start', icon=folium.Icon(color=color)).add_to(map)
        folium.Marker(location=coordinate[-1], tooltip=name + ' end', icon=folium.Icon(color=color)).add_to(map)
    if mode == 'atlantic':
        map.save('atlantic-map.html')
    elif mode == 'pacific':
        map.save('pacific-map.html')


def single_step_visualization(hurricane_df):
    DATA_FILE = 'data/single.txt'
    DATA_FILE_2 = 'data/single_2.txt'

    id = ''
    name = None
    actual = []
    predict = []
    actual_pattern = re.compile('^   actual')
    predict_pattern = re.compile('^   predict')
    lat_long_pattern = re.compile('\(.*?\)')
    name_pattern = re.compile('^Name')
    id_pattern = re.compile('^ID')
    map = folium.Map(zoom_start=6)
    valid = ['EMA', 'FAUSTO', 'HENRIETTE', 'MARIE', 'WALAKA']
    valid_2 = ['MARIO', 'TWENTYONE', 'HERNAN', 'NATE', 'ALBERTO', 'MICHAEL', 'HUMBERTO', 'CRISTOBAL', 'ISAIAS', 'OMAR',
                'GAMMA', 'DELTA', 'ZETA']

    with open(DATA_FILE_2, 'r') as f:
        last_id = None
        for line in f:
            line = line.rstrip()
            for match in re.finditer(id_pattern, line):
                id = int(line.split(' ')[-1])
                if id <= 48:
                    id += 1121
                else:
                    id += 1799
                if last_id != id and name in valid_2:
                    actual = hurricane_df.loc[(hurricane_df['ID'] == int(last_id)) & (hurricane_df['NAME'] == name)][['LAT', 'LONG']].values.tolist()[0:12]
                    predict_first_five = hurricane_df.loc[(hurricane_df['ID'] == int(last_id)) & (hurricane_df['NAME'] == name)][['LAT', 'LONG']].values.tolist()[0:5]
                    predict_first_five.extend(predict)
                    predict_PolyLine = folium.PolyLine(locations=predict_first_five, weight=3, color='red', tooltip='Predicted '+ name, id=name)
                    map.add_child(predict_PolyLine)
                    actual_PolyLine = folium.PolyLine(locations=actual, weight=3, color='blue', tooltip='Actual '+ name, id=name)
                    map.add_child(actual_PolyLine)
                    folium.Marker(location=actual[0], tooltip=name + ' start', icon=folium.Icon(color='green')).add_to(map)
                    folium.Marker(location=predict_first_five[-1], tooltip=name + ' end predicted', icon=folium.Icon(color='red')).add_to(map)
                    folium.Marker(location=actual[-1], tooltip=name + ' end actual', icon=folium.Icon(color='blue')).add_to(map)
                actual = []
                predict = []
                last_id = id
            for match in re.finditer(name_pattern, line):
                name = line.split(' ')[-1]
            for match in re.finditer(actual_pattern, line):
                temp = lat_long_pattern.findall(line)[0][1:-1]
                if temp:
                    temp = temp.split(', ')
                    lat = temp[0]
                    long = temp[1]
                    actual.append([float(lat), float(long)])
            for match in re.finditer(predict_pattern, line):
                temp = lat_long_pattern.findall(line)[0][1:-1]
                if temp:
                    temp = temp.split(', ')
                    lat = temp[0]
                    long = temp[1]
                    predict.append([float(lat), float(long)])
    map.save('prediction.html')


def multiple_step_visualization():
    pass


if __name__ == "__main__":
    ATLANTIC_DATA = 'atlantic_cleanned_hurdat2.csv'
    PACIFIC_DATA = 'pacific_cleanned_hurdat2.csv'

    # getting the data
    get_data = False
    if get_data:
        prepare_header(ATLANTIC_DATA)
        hurricane_data = data_extraction(ATLANTIC_DATA)

        prepare_header(PACIFIC_DATA)
        hurricane_data = data_extraction(PACIFIC_DATA, 'pacific')

    # load data
    atlantic_hurricane_df = pd.read_csv(ATLANTIC_DATA, sep=",")
    pacific_hurricane_df = pd.read_csv(PACIFIC_DATA, sep=",")

    # print(atlantic_hurricane_df.head())
    # print(pacific_hurricane_df.head())

    # simple visualizations
    # graph_hurricane_category(atlantic_hurricane_df)
    # graph_hurricane_category(pacific_hurricane_df, mode='pacific')

    # graph_hurricane_month(atlantic_hurricane_df)
    # graph_hurricane_month(pacific_hurricane_df, mode='pacific')

    # storm_track(atlantic_hurricane_df)
    # storm_track(pacific_hurricane_df, 'pacific')
    both_df = pd.concat([atlantic_hurricane_df, pacific_hurricane_df], axis=0)
    single_step_visualization(both_df)
