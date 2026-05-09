
import numpy as np
import cv2



# draw an given annotation cor/tra into an aimge
def drawAnnImg (oimg, curAnnotation, lineColor, lineColorB = None, lineThickness = 2):
    curAnnotation = curAnnotation.iloc[0]
    ofs = 0
    _ = cv2.line(oimg, (int(curAnnotation["cor_LL_x"])-ofs, int(curAnnotation["cor_LL_y"])-ofs), (int(curAnnotation["cor_LR_x"])-ofs, int(curAnnotation["cor_LR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_LL_x"])-ofs, int(curAnnotation["cor_LL_y"])-ofs), (int(curAnnotation["cor_UL_x"])-ofs, int(curAnnotation["cor_UL_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_LR_x"])-ofs, int(curAnnotation["cor_LR_y"])-ofs), (int(curAnnotation["cor_UR_x"])-ofs, int(curAnnotation["cor_UR_y"])-ofs), color = lineColor, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["cor_UR_x"])-ofs, int(curAnnotation["cor_UR_y"])-ofs), (int(curAnnotation["cor_UL_x"])-ofs, int(curAnnotation["cor_UL_y"])-ofs), color = lineColor, thickness = lineThickness)

    if lineColorB is None:
        lineColorB = [l>>2 for l in lineColor]
    _ = cv2.line(oimg, (int(curAnnotation["tra_LL_x"])-ofs, int(curAnnotation["tra_LL_y"])-ofs), (int(curAnnotation["tra_LR_x"])-ofs, int(curAnnotation["tra_LR_y"])-ofs), color = lineColorB, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_LL_x"])-ofs, int(curAnnotation["tra_LL_y"])-ofs), (int(curAnnotation["tra_UL_x"])-ofs, int(curAnnotation["tra_UL_y"])-ofs), color = lineColorB, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_LR_x"])-ofs, int(curAnnotation["tra_LR_y"])-ofs), (int(curAnnotation["tra_UR_x"])-ofs, int(curAnnotation["tra_UR_y"])-ofs), color = lineColorB, thickness = lineThickness)
    _ = cv2.line(oimg, (int(curAnnotation["tra_UR_x"])-ofs, int(curAnnotation["tra_UR_y"])-ofs), (int(curAnnotation["tra_UL_x"])-ofs, int(curAnnotation["tra_UL_y"])-ofs), color = lineColorB, thickness = lineThickness)
    return (oimg)



def pseudoRGB (img, method = "clahe"):
    if method not in ["clahe"]:
        exit ("Pseudo RGB method " + str(method) + " is unknown.")

    conversionFactor = 256
    if img.dtype == np.uint8:
        conversionFactor  = 1
        method = 'clahe'

    if method == "clahe":
        factor = 0.5
        clipfactor = 2
        baseFactor = 16.0
        spreadFactor = 2.0

        clahe = cv2.createCLAHE(clipLimit=baseFactor*spreadFactor*clipfactor, tileGridSize=(int(2*factor),int(2*factor)))
        red = clahe.apply(img)
        clahe = cv2.createCLAHE(clipLimit=baseFactor*1/spreadFactor*clipfactor, tileGridSize=(int(8*factor),int(8*factor)))
        blue = clahe.apply(img)
        clahe = cv2.createCLAHE(clipLimit=baseFactor*clipfactor, tileGridSize=(int(4*factor),int(4*factor)))
        green = clahe.apply(img)

    img = cv2.merge((blue, green, red))
    return img



def getMontage (mrImg, patFrames = None, frame_size = 6, resize_size = 160, border_size = 32, frame_offset = None):
    middle_frame_index = mrImg.shape[0] // 2

    # Calculate the frame range
    start_frame = max(0, middle_frame_index - frame_size**2 // 2)
    end_frame = min(mrImg.shape[0], middle_frame_index + frame_size**2 // 2 + 1)
    frame_count = end_frame - start_frame

    montage_height = frame_size * resize_size + border_size*(frame_size+1)
    montage_width = frame_size * resize_size + border_size*(frame_size+1)
    montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)+128

    # Calculate the number of black frames to add on each side
    if frame_offset is None:
        frame_offset = max(0, (frame_size**2 - frame_count))
        frame_offset = np.random.randint(0,frame_offset) # do not always start at 0

    frame_idx = 0

    # in case we have more frames, we skip the first and last slices
    if frame_count > frame_size**2:
        frame_idx = (frame_count - frame_size**2)//2

    midFrame = None
    if patFrames is not None:
        # midframe currently not used
        midFrame = sorted(patFrames["curFrame"].values)
        midFrame = midFrame[len(midFrame)//2]
        exportAnnotation = True
    else:
        exportAnnotation = False
        boxes = None


    bboxes = ''
    for i in range(frame_size**2):
        if i < frame_offset or i >= frame_offset + frame_count:
            # Add black frames
            continue
        frame = mrImg[start_frame + frame_idx, :, :]
        resized_frame = cv2.resize(frame, (resize_size, resize_size))
        resized_frame = pseudoRGB (resized_frame)
        row = i % frame_size
        col = i // frame_size
        rpos = row*resize_size+(row+1)*border_size
        cpos = col*resize_size+(col+1)*border_size
        montage[cpos:cpos+resize_size, rpos:rpos+resize_size, :] = resized_frame

        if exportAnnotation == True:
            if midFrame is not None:
                ann = patFrames.query("curFrame == @frame_idx")
                ann = ann.query("curFrame == @midFrame")
            else:
                ann = patFrames.query("curFrame == @frame_idx")
            if len(ann) > 0:
                boxmargin = bm = 0
                y0, x0 = cpos+bm, rpos+bm
                y1, x1 = cpos+bm, rpos+resize_size-bm
                y2, x2 = cpos+resize_size-bm, rpos+resize_size-bm
                y3, x3 = cpos+resize_size-bm, rpos+bm
                bboxes += "{} {} {} {} {} {} {} {} {} 0\n".format(x0, y0, x1, y1, x2, y2, x3, y3, "box")

        frame_idx += 1
    return montage, bboxes



#
