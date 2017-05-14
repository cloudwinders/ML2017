import sys
import numpy as np 
from PIL import Image as img
import matplotlib.pyplot as plt
from scipy import linalg
import os

base_dir = os.path.dirname(os.path.realpath(__file__))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

class PCA:
    def __init__(self, pathname, count_num, char_count_num):
        trainData = []
        idx = "A"
        count = 0
        char_count = 1
        for file in sorted(os.listdir(pathname)):
            if file.endswith(".bmp"):
                if (file.startswith(idx) and (count < count_num)):
                    filepath = os.path.join(pathname, file)
                    imgArr = np.asarray(img.open(filepath))
                    imgList = np.reshape(imgArr, (np.product(imgArr.shape), )).astype('int')
                    trainData.append(imgList)
                    count += 1
                elif ((char_count < char_count_num) and (count == count_num)):
                    count = 0
                    char_count += 1
                    idx = chr(ord(idx)+1)
        trainData = np.asarray(trainData)
        self.trainData = trainData

    def get_orig_data(self):
        trainData = np.copy((self.trainData).reshape(self.trainData.shape[0],64,64))
        return trainData

    def column(self, matrix, i):
        return [row[i] for row in matrix]
    
    def findMean(self):
        dataMean =[]
        trainData = self.trainData
        for idx in np.arange(len(trainData[0])):
            meanValue = np.mean(self.column(trainData,idx))
            dataMean.append(meanValue)
        dataMean = np.asarray(dataMean)
        self.dataMean = dataMean
        return dataMean

    def setCenter(self):
        dataMean = self.dataMean
        trainData = self.trainData
        for idx in np.arange(len(self.column(trainData,0))):
            trainData[idx] = trainData[idx] - dataMean
        dataAdjust = np.asarray(trainData)
        return dataAdjust

    def SVD(self, matrix):
        [u,s,v] = linalg.svd(matrix)
        self.u = u
        self.s = s
        self.v = v
        return u,s,v

    def reduceDim(self, topEigenNum):
        s_red = []
        s = self.s
        for idx in np.arange(len(s)):
            if (idx < topEigenNum ):
                s_red.append(s[idx])
            else:
                s_red.append(0)
        s_red = np.asarray(s_red)
        S_red = linalg.diagsvd(s_red, 4096, 100)
        return S_red

    def reconData(self, u, s, v, mean):
        reconSet =[]
        recon = np.dot(np.dot(u,s),v)
        recon_t = recon.T
        for idx in np.arange(len(recon_t)):
            reconSet.append(recon_t[idx]+mean)
        reconSet = np.asarray(reconSet)
        return reconSet

    def uncenterData(self, matrix, mean):
        reconSet =[]
        for idx in np.arange(len(matrix)):
            reconSet.append(matrix[idx]+mean)
        reconSet = np.asarray(reconSet)
        return reconSet

    def saveImg(self, output, inputArr):
        size = inputArr.shape[0]
        num = int(np.sqrt(size))
        imgArr = np.asarray(inputArr).reshape(len(inputArr),64,64)
        plt.figure()
        for idx in np.arange(size):
            plt.subplot(num,num,idx+1)
            plt.imshow(imgArr[idx],cmap='gray')
            plt.axis('off')
        plt.savefig(os.path.join(img_dir, output))

    def averageFace(self, meanList):
        averageFace =np.asarray(meanList).reshape(64,64)
        plt.figure()
        plt.imshow(averageFace, cmap='gray')
        plt.savefig(os.path.join(img_dir, "averageFace.png"))

    def findEigenFace(self, u, s, vectorNum, mean):
        eigenFace = []
        temp = np.dot(u,s)
        temp = temp.T
        for idx in np.arange(vectorNum):
            eigenFace.append(temp[idx]+mean)
        eigenFace = np.asarray(eigenFace)
        return eigenFace

    def find_error(self, data, meanList):
        errorList =[]
        for idx in np.arange(100):
            s_red = pca_p1.reduceDim(idx+1)
            eigenFace = pca_p1.findEigenFace(self.u, s_red, idx+1, meanList)
            recon = pca_p1.reconData(self.u, s_red, self.v, meanList)
            recon = recon.astype(np.float32)

            error = np.sqrt(((data - recon) ** 2).mean()) / 256*100
            print("idx: %d, error: %f" %(idx, error))
            errorList.append(error)

            if(error<0.90):
                break

        errorList = np.asarray(errorList).astype(np.float32)

        plt.figure()
        plt.plot(errorList)
        plt.xlabel('dim')
        plt.ylabel('error')
        plt.savefig(os.path.join(img_dir, "error_graph.jpg"))

        with open(os.path.join(img_dir, "error_history"),'a') as f:
            for i in range(len(errorList)):
                f.write('{},{}\n'.format(i,errorList[i]))
    
if __name__ == "__main__":
    pca_p1 = PCA(sys.argv[1], 10, 10)

    #plot the origin image
    oldData = pca_p1.get_orig_data()
    pca_p1.saveImg("origin_100.png", oldData)

    meanList = pca_p1.findMean()

    #Plot the average face
    pca_p1.averageFace(meanList)

    dataAdjust = pca_p1.setCenter()
    data = pca_p1.uncenterData(dataAdjust, meanList)

    dataAdjust_t = dataAdjust.T
    u,s,v = pca_p1.SVD(dataAdjust_t)

    s_red = pca_p1.reduceDim(9)
    eigenFace = pca_p1.findEigenFace(u, s_red,9, meanList)

    #Plot the eigenface
    first_9_eigenFace = eigenFace[:9]
    pca_p1.saveImg("eigenFace_9",first_9_eigenFace)

    recon_s_red = pca_p1.reduceDim(5)
    recon = pca_p1.reconData(u, recon_s_red, v, meanList)

    #Plot the reconstruct face
    pca_p1.saveImg("reconstruct_100.png", recon)
    #find minimum error
    pca_p1.find_error(data, meanList)
