import math
import numpy as np

class MotionMetrics:
        # Motion Metric Calculators
    def calculate_time(self, timeStamps):
        if len(timeStamps) > 1:
            result = timeStamps[-1] - timeStamps[0]
        else:
            result = 0  # or handle the case appropriately, maybe setting result to 0 or some default value

        result = round(result*10)/10
        return result

    def calculate_path_length(self, Tvector):
        result = 0
        for i in range(1,len(Tvector)):
            result = result + math.sqrt((Tvector[i][0][3]-Tvector[i - 1][0][3])**2+(Tvector[i][1][3]-Tvector[i - 1][1][3])**2+(Tvector[i][2][3]-Tvector[i - 1][2][3])**2)
        result = result/1000
        result = round(result*100)/100
        return result

    def calculate_angular_length(self, Tvector):
        resultXY=0
        for i in range(1,len(Tvector)):
            temp=self.calculateAngleDistance(Tvector[i],Tvector[i - 1])
            resultXY=resultXY + math.sqrt(temp[0]**2+temp[1]**2)
        return resultXY

    def calculateAngleDistance(self, T1,T2):
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

    def calculate_response_orientation(self, Tvector):
        result=0
        for i in range(1,len(Tvector)):
            temp=self.calculateAngleDistance(Tvector[i],Tvector[i - 1])
            result=result + math.fabs(temp[2])
        return result

    def calculate_depth_perception(self, Tvector):
        result = 0
        for i in range(1,len(Tvector)):
            result = result + math.fabs((Tvector[i][2][3]-Tvector[i - 1][2][3]))
        return result/1000

    def calculate_motion_smoothness(self, Tvector, timeStamps):
        if len(timeStamps) < 1:
            return 0
        
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

    def calculateVelocity(self, Tvector,timeStamps):
        velocity = []
        for i in range(1,len(Tvector)):
            distance = math.sqrt((Tvector[i][0][3]-Tvector[i - 1][0][3])**2+(Tvector[i][1][3]-Tvector[i - 1][1][3])**2+(Tvector[i][2][3]-Tvector[i - 1][2][3])**2)
            deltaT = timeStamps[i]-timeStamps[i-1]
            velocity.append(distance / deltaT * 1000)
        return velocity # mm/s

    def calculate_average_velocity(self, Tvector,timeStamps):
        if len(timeStamps) < 1:
            return 0
        meanVelocity = np.mean(self.calculateVelocity(Tvector,timeStamps))
        return meanVelocity
