import pandas as pd
import numpy as np
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==================================================================================== #
# HELPER CLASSES
# ==================================================================================== #
class Position:
    def __init__(self, _latd, _long):
        self.latitude   = _latd
        self.longtitude = _long
        self.max_wind   = 0
        self.pressure   = 0
    
    def print(self):
        print("(%d, %d, %d, %d)" % (self.latitude, self.longtitude, self.max_wind, self.pressure))

    def vectorize(self):
        return [self.latitude, self.longtitude, self.max_wind, self.pressure]


class Storm:
    def __init__(self, _id):
        self.id            = _id
        self.path          = []             # list of visited positions
        self.genesis       = Position(0,0)  # original position
        self.name          = ""
        self.num_cat_flips = 0              # times the storm changes its category
        self.categories    = []

    def process(self):
        self.num_cat_flips = len(np.unique(self.categories))-1

    def print(self):
        print("ID:         ", self.id)
        print("Name:       ", self.name)
        print("path_length:", len(self.path))
        print("origin:     ", end=" ")
        self.genesis.print()
        print("num_flips:  ", self.num_cat_flips)
        print()



# ==================================================================================== #
# HELPER FUNCTIONS
# ==================================================================================== #
def build_storm_database(dframe, timesteps):
    storm_list = []
    storm_id   = 0

    for index in range(dframe.shape[0]):
        if dframe["ID"][index].astype(int) == storm_id:
            gen_pt          = Position(dframe["LAT"][index].astype(float), dframe["LONG"][index].astype(float))
            gen_pt.max_wind = dframe["MAX WIND"][index].astype(int)
            gen_pt.pressure = dframe["MIN PRESSURE"][index].astype(int)

            new_storm         = Storm(storm_id)
            new_storm.name    = dframe["NAME"][index]
            new_storm.genesis = gen_pt
            new_storm.categories.append(dframe["CATEGORY"][index].astype(int))
            new_storm.path.append(new_storm.genesis)
            storm_list.append(new_storm)
            storm_id += 1

        else:
            new_pt          = Position(dframe["LAT"][index].astype(float), dframe["LONG"][index].astype(float))
            new_pt.max_wind = dframe["MAX WIND"][index].astype(int)
            new_pt.pressure = dframe["MIN PRESSURE"][index].astype(int)

            storm_list[-1].path.append(new_pt)
            storm_list[-1].categories.append(dframe["CATEGORY"][index].astype(int))

    filtered_storm_list = []
    for entry in storm_list:
        if len(entry.path) > timesteps:
            filtered_storm_list.append(entry)

    return filtered_storm_list


def create_model(timesteps, inputs):
    model = Sequential()
    model.add(LSTM(64, activation="relu", input_shape=(timesteps, inputs, ), return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(32))
    model.add(Dropout(0.15))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


def generate_samples(database, timesteps, model_name):
    x_data = []
    y_data = []

    for entry in database:           
        for idx in range(0, len(entry.path)-timesteps):
            x_sample = []
            y_sample = []

            if model_name == "LATITUDE":
                y_sample.append(entry.path[idx+timesteps].latitude)
            else:
                y_sample.append(entry.path[idx+timesteps].longtitude)

            for t in range(timesteps):
                x_sample.append([entry.path[idx+t].latitude, entry.path[idx+t].longtitude, entry.path[idx+t].max_wind, entry.path[idx+t].pressure])
        
            x_data.append(x_sample)
            y_data.append(y_sample)

    return np.array(x_data), np.array(y_data)


def evaluate(latd_model, long_model, test_db, timesteps):
    # generate testing samples formatted for LSTM
    for entry in test_db:
        x_test_latd, y_test_latd = generate_samples([entry], timesteps, "LATITUDE")
        x_test_long, y_test_long = generate_samples([entry], timesteps, "LONGTITUDE")

        for index in range(len(x_test_latd)):
            latd_pred = latd_model.predict(x_test_latd[0])
            long_pred = long_model.predict(x_test_long[0])
            print("ID:", entry.id)
            print("actual:  (%f, %f)" % (y_test_latd[index][0], y_test_long[index][0]))
            print("predict: (%f, %f)" % (latd_pred[index][0], long_pred[index][0]))
            print("============================================================")

    
# ==================================================================================== #
# MAIN FUNCTION
# ==================================================================================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_steps", required=True, type=int)
    parser.add_argument("-f", "--infile", required=True, type=str)
    args = parser.parse_args()

    # build the complete storm database
    dframe = pd.read_csv(args.infile)
    database = build_storm_database(dframe, args.num_steps)

    # split the database for training/testing (90/10)
    split_pt = int(19*len(database)/20)
    
    train_db = database[:split_pt]
    test_db  = database[split_pt:]

    # generate training samples formatted for LSTM
    x_train_latd, y_train_latd = generate_samples(train_db, args.num_steps, "LATITUDE")
    x_train_long, y_train_long = generate_samples(train_db, args.num_steps, "LONGTITUDE")

    # create two models: one for latitude and one for longtitude 
    latd_model = create_model(x_train_latd.shape[1], x_train_latd.shape[2])
    long_model = create_model(x_train_long.shape[1], x_train_long.shape[2])

    # create checkpoint file
    latd_save = ModelCheckpoint('latd-weights/weights-{epoch:02d}-{loss:.4f}.hdf5', save_best_only=True, monitor='loss', mode='min')
    long_save = ModelCheckpoint('long-weights/weights-{epoch:02d}-{loss:.4f}.hdf5', save_best_only=True, monitor='loss', mode='min')

    # train latitude and longtitude models
    latd_model.fit(x_train_latd, y_train_latd, epochs=1, batch_size=64, callbacks=latd_save, verbose=False)
    long_model.fit(x_train_long, y_train_long, epochs=1, batch_size=64, callbacks=long_save, verbose=False)

    # evaluate models' performance on test sets
    print("MODEL PERFORMANCE")
    evaluate(latd_model, long_model, test_db, args.num_steps)


if __name__ == "__main__":
    main()
