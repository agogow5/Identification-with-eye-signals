import matplotlib as mpl
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import csv
import os
import cv2
import struct

def getDVSeventsDavis(file, ROI=np.array([]), numEvents=1e10, startEvent=0):
    print('\ngetDVSeventsDavis function called \n')
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY
    if len(ROI) != 0:
        if len(ROI) == 4:
            print('Region of interest specified')
            x0 = ROI(0)
            y0 = ROI(1)
            x1 = ROI(2)
            y1 = ROI(3)
        else:
            print(
                'Unknown ROI argument. Call function as: \n getDVSeventsDavis(file, ROI=[x0, y0, x1, y1], numEvents=nE, startEvent=sE) '
                'to specify ROI or\n getDVSeventsDavis(file, numEvents=nE, startEvent=sE) to not specify ROI')
            return

    else:
        print('No region of interest specified, reading in entire spatial area of sensor')

    print('Reading in at most', str(numEvents))
    print('Starting reading from event', str(startEvent))

    triggerevent = int('400', 16)
    polmask = int('800', 16)
    xmask = int('003FF000', 16)
    ymask = int('7FC00000', 16)
    typemask = int('80000000', 16)
    typedvs = int('00', 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []  # Timestamps tick is 1 us
    pol = []
    numeventsread = 0

    length = 0
    aerdatafh = open(file, 'rb')
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    print("file size", length)

    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    while p < length:
        ad, tm = struct.unpack_from('>II', tmp)
        ad = abs(ad)
        if (ad & typemask) == typedvs:
            xo = sizeX - 1 - float((ad & xmask) >> xshift)
            yo = float((ad & ymask) >> yshift)
            polo = 1 - float((ad & polmask) >> polshift)
            if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                x.append(xo)
                y.append(yo)
                pol.append(polo)
                ts.append(tm)
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1

    ts[:] = [x - ts[0] for x in ts]  # absolute time -> relative time
    x[:] = [int(a) for a in x]
    y[:] = [int(a) for a in y]

    print('Total number of events read =', numeventsread)
    print('Total number of DVS events returned =', len(ts))

    return ts, x, y, pol

def event_neighbor_filter(data=np.array([[]]), height=260, width=346, margin=2, threshold=2):
    img = np.zeros([height, width], dtype=np.int8)
    
    for idx in range(0, data.shape[0]):
        if(data[idx, 2] == 1):
            img[int(data[idx, 1]), int(data[idx, 0])] = 1    #pol = 1
        else:
            img[int(data[idx, 1]), int(data[idx, 0])] = -1  #pol = 0
            
    pos_tuple = np.where((img == 1) | (img == -1))
    pos = np.array([pos_tuple[0], pos_tuple[1]]).T

    img_padding = np.zeros([height + 2 * margin, width + 2 * margin], dtype=np.int8)
    img_padding[margin:height + margin, margin:width + margin] = img
    
    
    for idx in range(0, pos.shape[0]):
        num_of_events = 0
        for i in range(-margin, margin + 1):
            for j in range(-margin, margin + 1):
                num_of_events += abs(img_padding[pos[idx][0] + i][pos[idx][1] + j])
        is_event = num_of_events > threshold
        if(is_event == False):
            img[pos[idx][0]][pos[idx][1]] = 0

    data_filtered_tuple = np.where((img == 1) | (img == -1))
    img_tuple = np.array(img[:]).reshape(-1,1)
    img_filtered_tuple = img_tuple[(img_tuple == 1) | (img_tuple == -1)]
    data_filtered = np.array([data_filtered_tuple[1], data_filtered_tuple[0], img_filtered_tuple]).T
   
    return data_filtered

# ## Filterring Frame-based density

binary = False
filter_flag = False   # True : use filter
sub_num = 1        # subject number of filtration 
st_sub_idx = 5       # start subject index of filtration
sub_group = 1        # group number of each subject

date = '03-11'
# folder_name = 'G:/Software/DAVIS_data/' + date + '/'
folder_name = '../raw_data/dvs/'

# data_file = ['Davis346redColor-2019-' + date + 'Ts' + ("%02d" % i) +'-' 
#              + ("%02d" % j) +'.aedat'
#              for i in range(st_sub_idx, sub_num+st_sub_idx+1)
#                  for j in range(1, sub_group+1)
#             ]
data_file = ['Davis346redColor-2019-' + date + 'Ts' + ("%02d" % i) +'-' 
             +'em.aedat'
             for i in range(st_sub_idx, sub_num+st_sub_idx+1)]

csv_file = ['s'+("%02d" % i ) + '/' for i in range(st_sub_idx, sub_num+st_sub_idx+1)] 
img_file = ['em'+("%02d" % i ) + '/' for i in range(st_sub_idx, sub_num+st_sub_idx+1)]

head = ["density"]
file_name = "event_density_"

data_size = 70000   # 70s
X_pol = np.zeros((len(data_file), data_size), dtype = int)
X_neg = np.zeros((len(data_file), data_size), dtype = int)

for k in range(0, sub_num):
    for m in range(0, sub_group):
        step_time = 10_000  # 10000 us = 10 ms
        sliding_time = 2_000  # 2000 us = 2 ms   

        start_ts = 0      # 5,000,000 = 5s
        end_ts = 60000000 # 60,000,0000 = 60s

        print('loading file: ', folder_name + data_file[k*sub_group+m])
        
        ts, x, y, pol = getDVSeventsDavis(folder_name + data_file[k*sub_group+m])

        img = np.zeros((260, 346), dtype=np.uint8)  
        img_pol = np.zeros((260, 346, 3), dtype=np.uint8)    #RGB color

        start_idx = 0
        start_time = 0
        img_counter = 1

        X_pol[k*sub_group+m] = np.zeros((data_size), dtype=int)
        X_neg[k*sub_group+m] = np.zeros((data_size), dtype=int)

        # sort by timestamp
        raw_x = np.array(x[:]).reshape(-1,1)
        raw_y = np.array(y[:]).reshape(-1,1)
        raw_ts = np.array(ts[:]).reshape(-1,1)
        raw_pol = np.array(pol[:]).reshape(-1,1)
        raw_data = np.column_stack((raw_x, raw_y, raw_pol, raw_ts))
        index = np.lexsort(raw_data.T)
        raw_data = raw_data[index,:]

        #extract data from start_ts (30sec) 
        while ((start_time <= start_ts) & (start_time < raw_data[-1,3])):
            start_idx += 1
            start_time = raw_data[start_idx,3]

        end_idx = start_idx
        end_time = start_time + step_time

        # generate frames with filteration
        while ((end_time <= raw_data[-1,3]) & (end_time < end_ts)):    #raw_data[-1,3] = the last timestamp        
            while raw_data[end_idx,3] < end_time:
                end_idx += 1

            data = raw_data[start_idx:end_idx]

            pos_y_center = neg_y_center = 0   
            if filter_flag is True:
                data_filter = event_neighbor_filter(data, height=260, width=346, margin=2, threshold=8)   
                for i in range(0, data_filter.shape[0]):
                    if(data_filter[i][2] == 1):
                        if(binary == True):
                            img_pol[data_filter[i][1] - 1][data_filter[i][0] - 1] = [255,255,255]
                        else:
                            img_pol[data_filter[i][1] - 1][data_filter[i][0] - 1] = [152,251,152]
                        pos_y_center += data_filter[i][1] - 1
                        X_pol[k*sub_group+m][img_counter] += 1
                    else:
                        if(binary == True):
                            img_pol[data_filter[i][1] - 1][data_filter[i][0] - 1] = [255,255,255]
                        else:    
                            img_pol[data_filter[i][1] - 1][data_filter[i][0] - 1] = [34,34,178]
                        neg_y_center += data_filter[i][1] - 1
                        X_neg[k*sub_group+m][img_counter] += 1
                # X[k][img_counter] = data_filter.shape[0]              
            else:
                for i in range(0, data.shape[0]):
                    if(data[i][2] == 1):
                        if(binary == True):
                            img_pol[int(data[i][1] - 1)][int(data[i][0] - 1)] = [255,255,255]
                        else:
                            img_pol[int(data[i][1] - 1)][int(data[i][0] - 1)] = [152,251,152]
                        pos_y_center += data[i][1] - 1
                        X_pol[k*sub_group+m][img_counter] += 1
                    else:
                        if(binary == True):
                            img_pol[int(data[i][1] - 1)][int(data[i][0] - 1)] = [255,255,255]
                        else:
                            img_pol[int(data[i][1] - 1)][int(data[i][0] - 1)] = [34,34,178]
                        neg_y_center += data[i][1] - 1
                        X_neg[k*sub_group+m][img_counter] += 1
                # X[k][img_counter] = data.shape[0]  

            # if(pos_y_center < neg_y_center):
            #     is_closing = 1
            # else:
            #     is_closing = -1
            # if(X[k][img_counter] is not 0):
            #     X[k][img_counter] *= is_closing

            img_pol = cv2.flip(img_pol, 0)          # x-axis reverse
        #         cv2.imshow('dvs', img_pol)
        #         cv2.waitKey(5)
            imgFullFile = img_file[k] + ('%05d' % img_counter) + '.png'
            print(imgFullFile)
            cv2.imwrite(imgFullFile, img_pol)

            while raw_data[start_idx,3] - (start_time - start_time % sliding_time) < sliding_time:
                start_idx += 1                          # update start_idx
            start_time = raw_data[start_idx, 3]         # update start_time
            end_time = start_time + step_time           # update end_time

            img_pol[:] = [0,0,0]                        # clear image
            img_counter += 1                            # image number count

        print("data ",k*sub_group+m+1," filteration done.")
        # print('saved as: ', folder_name + csv_file[int(k)] + file_name + ('s%02d' % (k+st_sub_idx) ) + ('_%d' % (m+1)) +"_pol.csv")
        # print('saved as: ', folder_name + csv_file[int(k)] + file_name + ('s%02d' % (k+st_sub_idx) ) + ('_%d' % (m+1)) +"_neg.csv")
        
        # pol_densDF = pd.DataFrame(X_pol[k*sub_group+m], columns=head)
        # pol_densDF.to_csv(folder_name + csv_file[int(k)] + file_name + ('s%02d' % (k+st_sub_idx) ) + ('_%d' % (m+1)) +"_pol.csv")
        # neg_densDF = pd.DataFrame(X_neg[k*sub_group+m], columns=head)
        # neg_densDF.to_csv(folder_name + csv_file[int(k)] + file_name + ('s%02d' % (k+st_sub_idx) ) + ('_%d' % (m+1)) +"_neg.csv")