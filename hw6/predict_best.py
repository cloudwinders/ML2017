import os
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from movies_class import data_process

if __name__ == "__main__":
    movies_class = data_process()

    users_data = pd.read_csv(os.path.join(sys.argv[1],'users.csv'), sep='::', engine='python')
    movies_data = pd.read_csv(os.path.join(sys.argv[1],'movies.csv'), sep='::', engine='python')

    test_data = pd.read_csv(os.path.join(sys.argv[1],'test.csv'), sep=',', engine='python')
    test_movies = test_data.MovieID.values

    print("Data loading...")
    print('Users data: ', users_data.shape)
    print('Movies data: ', movies_data.shape)

    numUserGender = {"M":0, "F":1}
    users_data.Gender = users_data.Gender.replace(numUserGender)
    users_data.Gender = users_data.Gender.astype('int')

    #from progress.bar import Bar
    test_users = test_data.UserID.values
    print("Processing test %d User's Data..." %(len(test_users)))
    #bar = Bar('Processing', max=len(test_users))
    testUserDetailsInput = []
    for i in range(len(test_users)):
        testUserDetailsInput.append(users_data[users_data.UserID == test_users[i]].values[0][0:4])
        #bar.next()
    #bar.finish()
    testUserDetailsInput = np.asarray(testUserDetailsInput, dtype='int')
    print('Test User Input data: ', testUserDetailsInput.shape)

    model = load_model('best_model.h5', custom_objects={'root_mean_squared_error': movies_class.root_mean_squared_error})
    Y_test = model.predict([testUserDetailsInput, test_movies])
    
    test_output = test_data
    test_output['UserID'] = Y_test
    test_output = test_output.drop('MovieID', 1)
    test_output.to_csv(sys.argv[2], sep=',', header=['TestDataID', 'Rating'], index=False)
    