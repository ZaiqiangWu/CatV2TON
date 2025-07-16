import os
import numpy as np

class Image2VideoWriter():
    def __init__(self):
        self.image_list = []

    def append(self,image):
        self.image_list.append(image)

    def make_video(self,outvid=None, fps=5, size=None,
                   is_color=True, format="MP4V", isRGB=False):
        """
        Create a video from a list of images.

        @param      outvid      output video
        @param      images      list of images to use in the video, BGR format
        @param      fps         frame per second
        @param      size        size of each frame
        @param      is_color    color
        @param      format      see http://www.fourcc.org/codecs.php
        @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

        The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
        By default, the video will have the size of the first image.
        It will resize every image to this size before adding them to the video.
        """
        from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
        fourcc = VideoWriter_fourcc(*format)
        vid = None
        for image in self.image_list:
            img = image
            if isRGB:
                img=img[:,:,[2,1,0]]
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                    if size[0]+size[1]>3000:
                        size = img.shape[1]//2, img.shape[0]//2
                vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        path, name = os.path.split(outvid)
        os.system("ffmpeg -i " + outvid + " -vcodec libx264 " + os.path.join(path,name.split('.')[0]+'temp.mp4'))
        os.system("rm " + outvid)
        os.system("mv "+os.path.join(path, name.split('.')[0]+'temp.mp4')+ " "+outvid)
        return vid
