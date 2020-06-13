"""
Intro. to Deep Learnning Assignment #2
Tal Raveh
203342795

PART 2
"""

import numpy as np
import time

# Read a file named 'file_name' with number patterns and 
# builds a data structure containing the input-output examples.
def read_file(file_name):
    # Check the run time elapsed
    s_time = time.time()                        # start the timer
    rp_view = np.array([[],[],[]])
    # Open the file and start reading
    with open(file_name , "r") as file:
        count = 1                               # count the number of frames
        print("Loading patterns from" , file_name , "...")
        # Read each line
        for idx, line in enumerate(file):
            # Skips the header lines
            if (idx == count*6-2):
                long = float(line[121:130])
                lat = float(line[146:155])      # dji is wrong with their lat long
                alt = float(line[168:178])
                rp_view = np.hstack((rp_view , np.array([lat , long , alt] , ndmin=2).T))
                count += 1                  # advance to the next pattern
                    
        f_time = time.time()                    # end the timer
    print("Done reading %d frames in %.2f seconds" %(count,(f_time-s_time)))
    return rp_view

if __name__ == "__main__":
    rp_view = read_file('./view/circulation.srt').T
