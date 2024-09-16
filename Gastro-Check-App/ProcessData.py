import pandas as pd
import numpy as np
import math

# Load the CSV file into a DataFrame
filename = 'ProcedureData/tracked_motion_data_20240916_134927.csv'  # Update with your actual file path
df = pd.read_csv(filename, header=None, dtype=str)
# 'header=None' specifies that the CSV does not have a header row.
# 'dtype=str' ensures all data is read as strings.

# Function to parse each entry and convert it to a 4x4 matrix
def parse_matrix_entry(entry):
    """
    Convert a string entry representing a 4x4 matrix into a 4x4 numpy array.

    Args:
        entry (str): String representation of a 4x4 matrix.

    Returns:
        numpy.ndarray: A 4x4 matrix as a numpy array. If the input string is invalid,
        returns a 4x4 matrix of zeros.
    """
    # Remove unwanted characters from the entry string
    entry = entry.strip('[]').replace(']', ' ').replace('[', '').replace(']', '')
    
    if entry:
        # Convert the cleaned string to a numpy array of floats
        numbers = np.fromstring(entry, sep=' ', dtype=float)
        if numbers.size == 16:
            # Reshape into a 4x4 matrix if the number of elements is correct
            return numbers.reshape(4, 4)
        else:
            # Handle unexpected number of elements
            return np.zeros((4, 4))
    else:
        # Return an empty 4x4 matrix if the entry is empty
        return np.zeros((4, 4))

# Flatten the DataFrame to a 1D array of entries
flat_entries = df.values.flatten()

# Initialize lists to store matrices and timestamps
matrices_array = []
timestamps_array = []

# Iterate over flattened entries and separate matrices and timestamps
for i in range(0, len(flat_entries)):
    if i % 2:
        # Parse and append matrix entries
        matrices_array.append(parse_matrix_entry(flat_entries[i]))
    else:
        # Append timestamp entries
        timestamps_array.append(float(flat_entries[i]))

# Combine matrices and timestamps into a list of lists
combined_array = []
for i in range(0, len(timestamps_array)):
    # Append each matrix and its corresponding timestamp as a sublist
    combined_array.append([matrices_array[i], timestamps_array[i]])
            #combined_array[0][0] for translationmatrix first entry
            #combined_array[0][1] for timstamp first entry

# Print the first matrix for verification
#print(combined_array[0][0])


# Calculates the duration of the procedure (start of tracking, untill end of tracking in seconds)
def calculateTime(timeStamps):
    print(timeStamps[-1])
    print(timeStamps[0])
    result = timeStamps[-1] - timeStamps[0]
    result = round(result*10)/10
    return result

def calculatePathLength(Tvector):
    result = 0
    for i in range(1,len(Tvector)):
        result = result + math.sqrt((Tvector[i][0][3]-Tvector[i - 1][0][3])**2+(Tvector[i][1][3]-Tvector[i - 1][1][3])**2+(Tvector[i][2][3]-Tvector[i - 1][2][3])**2)
    result = result/1000
    result = round(result*100)/100
    return result


def calculateAngularLength(Tvector):
    resultXY=0
    for i in range(1,len(Tvector)):
        temp=calculateAngleDistance(Tvector[i],Tvector[i - 1])
        resultXY=resultXY + math.sqrt(temp[0]**2+temp[1]**2)
    return resultXY

def calculateAngleDistance(T1,T2):
    angles = [0,0,0]
    R11=T2[0][0] * T1[0][0] + T2[0][1] * T1[0][1] + T2[0][2] * T1[0][2]
    R21=T2[1][0] * T1[0][0] + T2[1][1] * T1[0][1] + T2[1][2] * T1[0][2]
    R31=T2[2][0] * T1[0][0] + T2[2][1] * T1[0][1] + T2[2][2] * T1[0][2]
    R32=T2[2][0] * T1[1][0] + T2[2][1] * T1[1][1] + T2[2][2] * T1[1][2]
    R33=T2[2][0] * T1[2][0] + T2[2][1] * T1[2][1] + T2[2][2] * T1[2][2]
    angles[0]=math.atan2(R21,R11)
    angles[1]=math.atan2(- R31,math.sqrt(R32 * R32 + R33 * R33))
    angles[2]=math.atan2(R32,R33)
    return angles

def calculateResponseOrientation(Tvector):
    result=0
    for i in range(1,len(Tvector)):
        temp=calculateAngleDistance(Tvector[i],Tvector[i - 1])
        result=result + math.fabs(temp[2])
    return result

def calculateDepthPerception(Tvector):
    result = 0
    for i in range(1,len(Tvector)):
        result = result + math.fabs((Tvector[i][2][3]-Tvector[i - 1][2][3]))
    return result/1000

def calculateMotionSmoothness(Tvector, timeStamps):
    T=timeStamps[-1];
    d1x_dt1=[]; d1y_dt1=[]; d1z_dt1=[]; deltaT = []; timeStampsNew = [];
    for i in range(1,len(Tvector)):
        deltaT.append(timeStamps[i]-timeStamps[i-1])
        d1x_dt1.append((Tvector[i][0][3] - Tvector[i-1][0][3]) / deltaT[i-1])
        d1y_dt1.append((Tvector[i][1][3] - Tvector[i-1][1][3]) / deltaT[i-1])
        d1z_dt1.append((Tvector[i][2][3] - Tvector[i-1][2][3]) / deltaT[i-1])
        timeStampsNew.append((timeStamps[i]+timeStamps[i-1])/2)
    timeStamps = timeStampsNew

    d2x_dt2=[]; d2y_dt2=[]; d2z_dt2=[]; deltaT = []; timeStampsNew = [];
    for i in range(1,len(d1x_dt1)):
        deltaT.append(timeStamps[i]-timeStamps[i-1])
        d2x_dt2.append((d1x_dt1[i] - d1x_dt1[i-1]) / deltaT[i-1])
        d2y_dt2.append((d1y_dt1[i] - d1y_dt1[i-1]) / deltaT[i-1])
        d2z_dt2.append((d1z_dt1[i] - d1z_dt1[i-1]) / deltaT[i-1])
        timeStampsNew.append((timeStamps[i]+timeStamps[i-1])/2)
    timeStamps = timeStampsNew

    d3x_dt3=[]; d3y_dt3=[]; d3z_dt3=[]; deltaT = []; timeStampsNew = [];
    for i in range(1,len(d2x_dt2)):
        deltaT.append(timeStamps[i]-timeStamps[i-1])
        d3x_dt3.append((d2x_dt2[i] - d2x_dt2[i-1]) / deltaT[i-1])
        d3y_dt3.append((d2y_dt2[i] - d2y_dt2[i-1]) / deltaT[i-1])
        d3z_dt3.append((d2z_dt2[i] - d2z_dt2[i-1]) / deltaT[i-1])
        timeStampsNew.append((timeStamps[i]+timeStamps[i-1])/2)
    timeStamps = timeStampsNew

    j = [(x**2 + y**2 +z**2) for x, y ,z in zip(d3x_dt3, d3y_dt3, d3z_dt3)]
    MS = math.sqrt((1 / (2*T)) * np.trapz(j,timeStamps))
    return MS*10**6     # m/s^3

def calculateVelocity(Tvector,timeStamps):
    velocity = []
    for i in range(1,len(Tvector)):
        distance = math.sqrt((Tvector[i][0][3]-Tvector[i - 1][0][3])**2+(Tvector[i][1][3]-Tvector[i - 1][1][3])**2+(Tvector[i][2][3]-Tvector[i - 1][2][3])**2)
        deltaT = timeStamps[i]-timeStamps[i-1]
        velocity.append(distance / deltaT * 1000)
    return velocity # mm/s

def averageVelocity(Tvector,timeStamps):
    meanVelocity = np.mean(calculateVelocity(Tvector,timeStamps))
    return meanVelocity

print(averageVelocity(matrices_array, timestamps_array))