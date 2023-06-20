problems = {'DriverIdentification': {},
            'ConfLongDemo_JSI': {},
            'Healthy_Older_People': {},
            'Motor_Failure_Time': {},
            'Power_consumption': {},
            'PRSA2017': {},
            'RSSI': {},
            'User_Identification_From_Walking': {},
            'WISDM': {},
            }

problems['DriverIdentification']['dataset'] = './datasets/DriverIdentification/'
problems['DriverIdentification']['n_classes'] = 10
problems['DriverIdentification']['features'] = ['x-accelerometer', 'y-accelerometer', 'z-accelerometer', 'x-gyroscope',
                                                'y-gyroscope', 'z-gyroscope']
problems['DriverIdentification']['sample_rate'] = 2
problems['DriverIdentification']['data_length_time'] = -1
problems['DriverIdentification']['n_h_block'] = 15
problems['DriverIdentification']['n_train_h_block'] = 9
problems['DriverIdentification']['n_valid_h_block'] = 2
problems['DriverIdentification']['n_test_h_block'] = 4
problems['DriverIdentification']['h_moving_step'] = 1
problems['DriverIdentification']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60,
                                                      6 * 60, 7 * 60, 8 * 60, 9 * 60, 10 * 60]
problems['DriverIdentification']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60,
                                                      6 * 60, 7 * 60]

problems['ConfLongDemo_JSI']['dataset'] = './datasets/ConfLongDemo_JSI/'
problems['ConfLongDemo_JSI']['n_classes'] = 5
problems['ConfLongDemo_JSI']['features'] = ["x", "y", "z"]
problems['ConfLongDemo_JSI']['sample_rate'] = 30
problems['ConfLongDemo_JSI']['data_length_time'] = -1
problems['ConfLongDemo_JSI']['n_h_block'] = 15
problems['ConfLongDemo_JSI']['n_train_h_block'] = 9
problems['ConfLongDemo_JSI']['n_valid_h_block'] = 2
problems['ConfLongDemo_JSI']['n_test_h_block'] = 4
problems['ConfLongDemo_JSI']['h_moving_step'] = 1
problems['ConfLongDemo_JSI']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60]
problems['ConfLongDemo_JSI']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60]

problems['Healthy_Older_People']['dataset'] = './datasets/Healthy_Older_People/'
problems['Healthy_Older_People']['n_classes'] = 12
problems['Healthy_Older_People']['features'] = ["X", "Y", "Z"]
problems['Healthy_Older_People']['sample_rate'] = 1
problems['Healthy_Older_People']['data_length_time'] = -1
problems['Healthy_Older_People']['n_h_block'] = 15
problems['Healthy_Older_People']['n_train_h_block'] = 9
problems['Healthy_Older_People']['n_valid_h_block'] = 2
problems['Healthy_Older_People']['n_test_h_block'] = 4
problems['Healthy_Older_People']['h_moving_step'] = 1
problems['Healthy_Older_People']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60]
problems['Healthy_Older_People']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60]

problems['Motor_Failure_Time']['dataset'] = './datasets/Motor_Failure_Time/'
problems['Motor_Failure_Time']['n_classes'] = 3
problems['Motor_Failure_Time']['features'] = ['x', 'y', 'z']
problems['Motor_Failure_Time']['sample_rate'] = 18
problems['Motor_Failure_Time']['data_length_time'] = -1
problems['Motor_Failure_Time']['n_h_block'] = 15
problems['Motor_Failure_Time']['n_train_h_block'] = 9
problems['Motor_Failure_Time']['n_valid_h_block'] = 2
problems['Motor_Failure_Time']['n_test_h_block'] = 4
problems['Motor_Failure_Time']['h_moving_step'] = 1
problems['Motor_Failure_Time']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60,
                                                    6 * 60, 7 * 60, 8 * 60, 9 * 60, 10 * 60]
problems['Motor_Failure_Time']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60,
                                                    6 * 60, 7 * 60]

problems['Power_consumption']['dataset'] = './datasets/Power_consumption/'
problems['Power_consumption']['n_classes'] = 3
problems['Power_consumption']['features'] = ['Temperature', 'Humidity', 'Wind Speed',
                                             'general diffuse flows', 'diffuse flows',
                                             'Consumption']
problems['Power_consumption']['sample_rate'] = 1
problems['Power_consumption']['data_length_time'] = -1
problems['Power_consumption']['n_h_block'] = 15
problems['Power_consumption']['n_train_h_block'] = 9
problems['Power_consumption']['n_valid_h_block'] = 2
problems['Power_consumption']['n_test_h_block'] = 4
problems['Power_consumption']['h_moving_step'] = 1
problems['Power_consumption']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60,
                                                   6 * 60, 7 * 60, 8 * 60, 9 * 60, 10 * 60, 20 * 60, 30 * 60, 40 * 60]
problems['Power_consumption']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60,
                                                   6 * 60, 7 * 60, 8 * 60, 9 * 60, 10 * 60, 20 * 60]

problems['PRSA2017']['dataset'] = './datasets/PRSA2017/'
problems['PRSA2017']['n_classes'] = 12
problems['PRSA2017']['features'] = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd',
                                    'WSPM']
problems['PRSA2017']['sample_rate'] = 1
problems['PRSA2017']['data_length_time'] = -1
problems['PRSA2017']['n_h_block'] = 15
problems['PRSA2017']['n_train_h_block'] = 9
problems['PRSA2017']['n_valid_h_block'] = 2
problems['PRSA2017']['n_test_h_block'] = 4
problems['PRSA2017']['h_moving_step'] = 1
problems['PRSA2017']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60, 6 * 60,
                                          7 * 60, 8 * 60, 9 * 60, 10 * 60, 20 * 60, 30 * 60, 40 * 60]
problems['PRSA2017']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60, 6 * 60,
                                          7 * 60, 8 * 60, 9 * 60, 10 * 60, 20 * 60]

problems['RSSI']['dataset'] = './datasets/RSSI/'
problems['RSSI']['n_classes'] = 12
problems['RSSI']['features'] = ['rssiOne', 'rssiTwo']
problems['RSSI']['sample_rate'] = 1
problems['RSSI']['data_length_time'] = -1
problems['RSSI']['n_h_block'] = 6
problems['RSSI']['n_train_h_block'] = 4
problems['RSSI']['n_valid_h_block'] = 1
problems['RSSI']['n_test_h_block'] = 1
problems['RSSI']['h_moving_step'] = 1
problems['RSSI']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60]
problems['RSSI']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60]

problems['User_Identification_From_Walking']['dataset'] = './datasets/User_Identification_From_Walking/'
problems['User_Identification_From_Walking']['n_classes'] = 13
problems['User_Identification_From_Walking']['features'] = [' x acceleration', ' y acceleration', ' z acceleration']
problems['User_Identification_From_Walking']['sample_rate'] = 32
problems['User_Identification_From_Walking']['data_length_time'] = -1
problems['User_Identification_From_Walking']['n_h_block'] = 10
problems['User_Identification_From_Walking']['n_train_h_block'] = 5
problems['User_Identification_From_Walking']['n_valid_h_block'] = 2
problems['User_Identification_From_Walking']['n_test_h_block'] = 3
problems['User_Identification_From_Walking']['h_moving_step'] = 1
problems['User_Identification_From_Walking']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                                                  18, 19, 20, 21, 22, 23, 24]
problems['User_Identification_From_Walking']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

problems['WISDM']['dataset'] = './datasets/WISDM/'
problems['WISDM']['n_classes'] = 10
problems['WISDM']['features'] = ['X-accel', 'Y-accel', 'Z-accel']
problems['WISDM']['sample_rate'] = 20
problems['WISDM']['data_length_time'] = -1
problems['WISDM']['n_h_block'] = 15
problems['WISDM']['n_train_h_block'] = 9
problems['WISDM']['n_valid_h_block'] = 4
problems['WISDM']['n_test_h_block'] = 2
problems['WISDM']['h_moving_step'] = 1
problems['WISDM']['decision_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60, 6 * 60, 7 * 60,
                                       8 * 60, 9 * 60, 10 * 60]
problems['WISDM']['segments_times'] = [3, 4, 5, 6, 7, 8, 9, 10, 30, 60, 2 * 60, 3 * 60, 4 * 60, 5 * 60, 6 * 60, 7 * 60]
