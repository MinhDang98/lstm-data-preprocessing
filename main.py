import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


def prepare_header(data):
    with open(data, 'w') as wf:
        wf.write('ID,NAME,YEAR,MONTH,LAT,LONG,MAX WIND,MIN PRESSURE,CATEGORY\n')

def data_extraction(data):
    ORIGINAL_DATA = 'original_hurdat2.txt'
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
            hurricane_id += 1
            # header line has length = 4
            if len(line) == 4:
                if line[1].replace(' ', '') == 'UNNAMED':
                    hurricane_name = 'UNNAMED_' + str(unnamed_counter)
                else:
                    hurricane_name = line[1].replace(' ', '')
                unnamed_counter += 1
            else:
                year = line[0][:4].replace(' ', '')
                month = line[0][4:6].replace(' ', '')
                lat = line[4].replace(' ', '')
                long = '-' + line[5].replace(' ', '')
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
                min_pressure = line[7].replace(' ', '')
                with open(data, 'a') as wf:
                    wf.write(str(hurricane_id) + ',' + hurricane_name + ',' +  year + ',' + month +
                            ',' + lat + ',' + long + ',' + str(max_wind) + ',' + min_pressure +
                            ',' + str(cat) + '\n')

def get_CO2(data):
    df = pd.read_csv(data)
    df = df.drop('unc', axis=1)
    return df

def graph_hurricane_category():
    grouped = hurricane_df.groupby('CATEGORY')['ID'].nunique()
    grouped.plot(kind='bar')

    ax = plot.gca()
    plot.xticks(rotation=0)
    plot.ylabel('Counts')
    plot.xlabel('Hurricane Category')
    plot.title('Distribution of Hurricane Categories in the Atlantic, Northeast and North Central Pacific Oceans')
    textstr = '\n'.join(('Category 0: Winds less than 74 mph',
                        'Category 1: Winds 74 to 95 mph',
                        'Category 2: Winds 96 to 110 mph',
                        'Category 3: Winds 111 to 130 mph',
                        'Category 4: Winds 131 to 155 mph',
                        'Category 5: Winds greater than 155 mph'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.7, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1.25), ha='center')
    plot.show()

def graph_hurricane_month():
    grouped = hurricane_df.groupby('MONTH')['ID'].nunique()
    grouped.plot(kind='bar')
    
    plot.xticks(rotation=0)
    plot.ylabel('Counts')
    plot.xlabel('Month')
    plot.title('Monthly Hurricane Count in the Atlantic and Northeast and North Central Pacific Oceans')
    plot.show()

def graph_co2_count(hur_df, co2_df):
    # only graph from 1959
    hurricane_by_year = hur_df.astype({"YEAR": int})
    hurricane_by_year = hurricane_by_year.drop_duplicates('NAME')
    hurricane_by_year = hurricane_by_year[hurricane_by_year['YEAR'] >= 1980]
    hurricane_by_year = hurricane_by_year.rename(str.lower, axis='columns')
    hurricane_by_year = hurricane_by_year.groupby('year').size()
    hurricane_by_year = hurricane_by_year.to_list()
    co2_df = co2_df[co2_df['year'] > 1980]
    co2_df = co2_df['mean'].to_list()
    co2_hurricane_df = pd.DataFrame({'hurricane count': hurricane_by_year,
                                    'co2 mean': co2_df})
    co2_hurricane_df['hurricane count'].plot(kind='bar')
    co2_hurricane_df['co2 mean'].plot(secondary_y=True)
    plot.title('Correlation of Mean CO2 level and Frequency of Hurricanes')
    ax = plot.gca()
    plot.show()

def reformat(data):
    pd_to_csv = pd.read_csv(data, sep=",")
    pd_to_csv.rename(columns={'ID': 'Serial_Num', 'MAX WIND': 'wmo_wind',
                        'MIN PRESSURE': 'wmo_pres', 'CATEGORY': 'Category',
                        'NAME': 'Name', 'LAT': 'Latitude', 'LONG': 'Longtitude',
                        'YEAR': 'year', 'MONTH': 'month'}, inplace=True)
    # pd_to_csv['Category'].replace({'DB': 1, 'LO': 1, 'SD': 2,
    #                             'EX': 2, 'SS': 3, 'TD': 3,
    #                             'TS': 4, 'HU': 5}, inplace=True)
    pd_to_csv['Longtitude'] = pd_to_csv['Longtitude'].str[:-1]
    pd_to_csv['Latitude'] = pd_to_csv['Latitude'].str[:-1]
    pd_to_csv['wmo_pres'] = pd_to_csv['wmo_pres'].astype(int)
    pd_to_csv['wmo_pres'].replace({-999: 0}, inplace=True)
    print(pd_to_csv.head())
    pd_to_csv.to_csv('data.csv', index=False)

if __name__ == "__main__":
    CLEAN_DATA = str('cleanned_hurdat2.txt')
    # getting the data
    get_data = False
    if get_data:
        prepare_header(CLEAN_DATA)
        hurricane_data = data_extraction(CLEAN_DATA)
        # reformat clean data
        reformat(CLEAN_DATA)

    # load data
    hurricane_df = pd.read_csv(CLEAN_DATA, sep=",")
    # print(hurricane_df.head())

    # load co2
    C02_DATA = 'co2.txt'
    co2_df = get_CO2(C02_DATA)
    # print(co2_df.head())

    # simple visualizations
    # graph_hurricane_category()
    # graph_hurricane_month()
    graph_co2_count(hurricane_df, co2_df)