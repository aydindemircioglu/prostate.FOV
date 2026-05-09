import shutil
import os


def recreatePath (path, dry = False):
    print ("Recreating path ", path)
    if dry == True:
        return None

    try:
        shutil.rmtree (path)
    except:
        pass
    os.makedirs (path)


def removePath (path, dry):
    print ("Removing path ", path)
    if dry == True:
        return None

    try:
        shutil.rmtree (path)
    except:
        pass



#
