# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:41:55 2025

@author: Admin
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

file_path = "C:/Users/David Teo/Desktop/Clinical Scores.xlsx"

df = pd.read_excel(file_path)
df.iloc[:,[0,2,3,4,5]]
df[df=='-'] = np.nan
p1 = df[df['Patient']==1].reset_index(drop=True)
p2 = df[df['Patient']==2].reset_index(drop=True)
p3 = df[df['Patient']==3].reset_index(drop=True)

fig,axs = plt.subplots(1,3)

axs[0].set_title('Spinal Cord Independence\nMeasure Mobility Subscale', fontsize=10)
axs[0].set_ylabel('Score (__/40)')
axs[0].set_xlabel('Months Post-op')
axs[0].set_xticks(range(0,8), [-1,1,2,3,4,5,6,7])
axs[0].plot(p1['Spinal Cord Independence Measure Mobility Subscale'], color = 'blue', label = 'P1')
axs[0].plot(p2['Spinal Cord Independence Measure Mobility Subscale'], color = 'orange', label = 'P2')
axs[0].plot(p3['Spinal Cord Independence Measure Mobility Subscale'], color = 'green', label = 'P3')

axs[1].set_title('Walking Index for\nSCI II Scale')
axs[1].set_ylabel('Score (__/20)')
axs[1].set_xlabel('Months Post-op')
axs[1].set_xticks(range(0,8), [-1,1,2,3,4,5,6,7])
axs[1].plot(p1['Walking Index for SCI II Scale'], color = 'blue', label = 'P1')
axs[1].plot(p2['Walking Index for SCI II Scale'], color = 'orange', label = 'P2')
axs[1].plot(p3['Walking Index for SCI II Scale'], color = 'green', label = 'P3')

axs[2].set_title('Truncal Assessment Scale', fontsize=10)
axs[2].set_ylabel('Score (__/44)')
axs[2].set_xlabel('Months Post-op')
axs[2].set_xticks(range(0,8), [-1,1,2,3,4,5,6,7])
axs[2].plot(p1['Truncal Assessment Scale'], color = 'blue', label = 'P1')
axs[2].plot(p2['Truncal Assessment Scale'], color = 'orange', label = 'P2')
axs[2].plot(p3['Truncal Assessment Scale'], color = 'green', label = 'P3')
axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()
plt.tight_layout()


fig,axs = plt.subplots(1,3)

axs[0].set_title('5x Sit-stand Test', fontsize=10)
axs[0].set_ylabel('Time (s)')
axs[0].set_xlabel('Months Post-op')
#axs[0].plot(p1['5x Sit-stand Test'], color = 'blue', label = 'P1')
#axs[0].plot(p2['5x Sit-stand Test'], color = 'orange', label = 'P2')
axs[0].plot(p3['5x Sit-stand Test'], color = 'green')
axs[0].set_xticks(range(0,8), [-1,1,2,3,4,5,6,7])

axs[1].set_title('10-metre walk test')
axs[1].set_ylabel('Time (s)')
axs[1].set_xlabel('Months Post-op')
axs[1].set_xticks(range(0,8), [-1,1,2,3,4,5,6,7])
#axs[1].plot(p1['10-metre walk test'], color = 'blue', label = 'P1')
#axs[1].plot(p2['10-metre walk test'], color = 'orange', label = 'P2')
axs[1].plot(p3['10-metre walk test'], color = 'green')

axs[2].set_title('2-min walk test', fontsize=10)
axs[2].set_ylabel('Speed (m/s)')
axs[2].set_xlabel('Months Post-op')
axs[2].set_xticks(range(0,8), [-1,1,2,3,4,5,6,7])
#axs[2].plot(p1['2-min walk test'], color = 'blue', label = 'P1')
#axs[2].plot(p2['2-min walk test'], color = 'orange', label = 'P2')
axs[2].plot(p3['2-min walk test'], color = 'green')

plt.legend()
plt.tight_layout()



