import os
import sys
import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import load_model
from movies_class import data_process

if __name__ == "__main__":
    movies_class = data_process()

    users_data = pd.read_csv(os.path.join(sys.argv[1],'users.csv'), sep='::', engine='python')
    movies_data = pd.read_csv(os.path.join(sys.argv[1],'movies.csv'), sep='::', engine='python')
    test_data = pd.read_csv(os.path.join(sys.argv[1],'test.csv'), sep=',', engine='python')

    test_users = test_data.UserID.astype('int')
    test_movies = test_data.MovieID.astype('int')

    print("Data loading...")
    print('Users data: ', users_data.shape)
    print('Movies data: ', movies_data.shape)

    model = load_model('model_MF.h5', custom_objects={'root_mean_squared_error': movies_class.root_mean_squared_error})
    Y_test = model.predict([test_users, test_movies])
    
    test_output = test_data
    test_output['UserID'] = Y_test
    test_output = test_output.drop('MovieID', 1)
    test_output.to_csv(sys.argv[2], sep=',', header=['TestDataID', 'Rating'], index=False)
    