def readdata(dataned1, flag, path, label):
    file = open(path)
    list_read = file.readlines()
    file.close()
    # dataned1 = []
    # flag = []
    for line in list_read:
        linestr = line.strip('[')
        linestr = linestr.strip('\n')
        linestr = linestr.strip(']')
        linestrlist = linestr.split(",")
        # linelist = map(int, linestrlist)
        linelist = [float(i) for i in linestrlist]
        dataned1.append(linelist)
        flag.append(label)
    return dataned1, flag