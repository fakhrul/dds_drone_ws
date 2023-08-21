import os
import re
from shutil import copyfile
from PIL import Image
import PIL

def is_good_image(filename):

    try:
        im = Image.open(filename)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        im.close() #reload is necessary in my case
        im = Image.open(filename) 
        # im.verify()
        im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        im.close()
        return True
    except Exception as e:
        print(e) 
        return False
    #manage excetions here


def main():

    source = 'C:/GitHub/dds_drone_ws/workspace/training_dds/images'
    print('Start cleaning corrupted images')
    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    for i in images:
        fullpath = source + "/" + i
        if is_good_image(fullpath) == False:
            print('BAD IMAGE', fullpath)
            os.remove(fullpath)
    print('End cleaning corrupted images')


    print('Start cleaning image no xml')
    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    for image_name in images:
        fullpath = source + "/" + image_name

        filenameWithouteExt = os.path.splitext(image_name)[0]
        fullpath_xml = source + "/" + filenameWithouteExt + ".xml"
        if os.path.exists(fullpath_xml) == False:
            print('No XML', fullpath_xml)
            os.remove(fullpath)

    print('End cleaning image no xml')

    print('Start cleaning image no images')
    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.xml)$', f)]

    for image_xml in images:
        fullpath = source + "/" + image_xml

        filenameWithouteExt = os.path.splitext(image_xml)[0]
        fullpath_png = source + "/" + filenameWithouteExt + ".png"
        if os.path.exists(fullpath_png) == False:
            print('No PNG', fullpath_png)
            os.remove(fullpath)

    print('End cleaning image no images')

if __name__ == '__main__':
    main()