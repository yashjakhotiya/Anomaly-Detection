import os
import shutil

def getAnomalyFrameRanges(fileName):
    anomalyFrameRanges = []
    with open(fileName) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 

    for line in range(1, len(content)):
        string = content[line]
        start = string.find('[') 
        end = string.find(']', start)
        anomalyFrameRanges.append(string[start:end + 1])
    return anomalyFrameRanges


def renameAndCopyTrainDataSet(folder, subfolder):
    mainDirName = "./" + folder + "/" + subfolder + "/Train/"
    renameImage = ""
    if(subfolder == "UCSDped1"):
        renameImage = "p1_"
    else:
        renameImage = "p2_"

    for dirname in os.listdir(mainDirName):
        absolutePath = mainDirName + dirname + "/"
        if os.path.isdir(absolutePath):
            for i, filename in enumerate(os.listdir(absolutePath)):
                #The train folder have data store files so ignore those files 
                if("Train" in dirname and "DS_Store" not in filename):
                    newImageName = renameImage + dirname + "_0_" + filename
                    destinationPath = "./Dataset/" + newImageName 
                    print(destinationPath)
                    shutil.copy2(absolutePath+filename, destinationPath)

def renameAndCopyTestDataSet(folder, subfolder):
    mainDirName = "./" + folder + "/" + subfolder + "/Test/"
    renameImage = ""
    pedFileName = ""
    if(subfolder == "UCSDped1"):
        renameImage = "p1_"
        pedFileName = "UCSDped1.m"
    else:
        renameImage = "p2_"
        pedFileName = "UCSDped2.m"

    anomalyFrameRanges = getAnomalyFrameRanges(mainDirName + pedFileName)
    dirCount = 0
    
    for dirname in os.listdir(mainDirName):    
        if("Test" in dirname and "_gt" not in dirname):
            absolutePath = mainDirName + dirname + "/"
            if os.path.isdir(absolutePath):
                frameRange = anomalyFrameRanges[dirCount]
                frameRange = frameRange[1:len(frameRange)-1]
                frameRangeList = frameRange.strip().split(",")

                for i, filename in enumerate(os.listdir(absolutePath)):
                    label = 0
                    for frames in frameRangeList:
                        frames = frames.strip().split(":")
                        if(i >= int(frames[0]) and i <= int(frames[1])):
                            label = 1
                            break 
                    
                    #The train folder have data store files so ignore those files 
                    if("DS_Store" not in filename):
                        pass
                        newImageName = renameImage + dirname + "_"+ str(label) +"_" + filename
                        destinationPath = "./Dataset/" + newImageName 
                        print(destinationPath)
                        shutil.copy2(absolutePath+filename, destinationPath)
            dirCount += 1
             

renameAndCopyTrainDataSet("UCSD_Anomaly_Dataset.v1p2", "UCSDped1")
renameAndCopyTrainDataSet("UCSD_Anomaly_Dataset.v1p2", "UCSDped2")
renameAndCopyTestDataSet("UCSD_Anomaly_Dataset.v1p2", "UCSDped1")
renameAndCopyTestDataSet("UCSD_Anomaly_Dataset.v1p2", "UCSDped2")