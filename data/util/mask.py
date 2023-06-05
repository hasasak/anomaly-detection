# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from random import sample
import cv2
import numpy as np
from PIL import Image, ImageDraw


def random_cropping_bbox(img_shape=(256,256), mask_mode='onedirection'):
    h, w = img_shape
    if mask_mode == 'onedirection':
        _type = np.random.randint(0, 4)
        if _type == 0:
            top, left, height, width = 0, 0, h, w//2
        elif _type == 1:
            top, left, height, width = 0, 0, h//2, w
        elif _type == 2:
            top, left, height, width = h//2, 0, h//2, w
        elif _type == 3:
            top, left, height, width = 0, w//2, h, w//2
    else:
        target_area = (h*w)//2
        width = np.random.randint(target_area//h, w)
        height = target_area//width
        if h==height:
            top = 0
        else:
            top = np.random.randint(0, h-height)
        if w==width:
            left = 0
        else:
            left = np.random.randint(0, w-width)
    return (top, left, height, width)


def test_bbox():
    scale=32
    list_patch_x_y=[]
    patch_sum = 256*256/32
    a = random.randint(0,15)
    b = random.randint(16,31)
    c = random.randint(32,47)
    d = random.randint(48,63)
    list_1 = [a,b,c,d]
    for i in list_1:
      top = (i //int(256/scale) )*scale
      left = (i % int(256/scale)) * scale
      height = int(scale)
      width = int(scale)
      top = int(top)
      left = int(left)
      list_patch_x_y.append((top, left, height, width))
    return list_patch_x_y
def all_bbox():
    bbox = []
    bbox_1 = [0,0,256,256]
    bbox.append(bbox_1)
    return bbox
def mask_bbox_all(img_shape,  dtype='uint8'):
    height, width = img_shape[:2]
    mask = np.ones((height, width, 1), dtype=dtype)
    return mask
def patch_bbox_random(type_1 =4):
    #patch_num = random.randint(0,1)
    #a = random.randint(0,1)
    #scale_range = [8,16,32]
    #scale = random.sample(scale_range,1)[0]
    #scale= 8
    #a= 0
    #patch_num=1
    #print(scale)
    #num = 256//scale/2
    #list_patch_x_y=[]
    #if a == 0:
      #if patch_num==0:
        #for i in range(int(num)):
          #top = 0
          #left = 0+i*scale*2
          #height = 256
          #width = int(scale)
          #list_patch_x_y.append((top, left, height, width))
      #else:
        #for i in range(int(num)):
          #top = 0
          #left = scale+i*scale*2
          #height = 256
          #width = int(scale)
          #list_patch_x_y.append((top, left, height, width))
    #if a== 1:
      #if patch_num==0:
        #for i in range(int(num)):
          #top = 0+i*scale*2
          #left = 0
          #height =int(scale)
          #width = 256
          #list_patch_x_y.append((top, left, height, width))
      #else:
        #for i in range(int(num)):
          #top = scale+i*scale*2
          #left = 0
          #height =int(scale)
          #width = 256
          #list_patch_x_y.append((top, left, height, width))
    #k = random.randint(0,3)
    #def is_odd(n):
      #return n % 4 == k

    #for i in filter(is_odd, list_all):
      #if i // (256/scale)%2==0:
        #a.append(i)
      #else:
        #a.append((i-(i//(256/scale))*(256/scale)+2)%(256/scale)+(i//(256/scale))*(256/scale))
    bbox = []
    bbox_1=[]
    bbox_2=[]
    bbox_3=[]
    bbox_4=[]
    if type_1==1:
      width = 256
      height = 4
      for i in range(32):
        print(i)
        top = i*8
        left=0
        bbox.append([top,left,height,width])
    if type_1==2:
      width = 256
      height = 4
      for i in range(32):
        print(i)
        top = i*8+4
        left=0
        bbox.append([top,left,height,width])
    if type_1 ==3:
      width =8
      height =8
      arg = np.arange(32*32)
      np.random.shuffle(arg)
      random_num=arg[0:32*8]
      random_num_1=arg[32*8:32*16]
      random_num_2=arg[32*16:32*24]
      random_num_3=arg[32*24:32*32]
    for i in random_num:
      top = int(i)//int(32)*8
      left = int(i)%int(32)*8
      bbox_1.append((top,left,height,width))
    
    for i in random_num_1:
      top = int(i)//int(32)*8
      left = int(i)%int(32)*8
      bbox_2.append((top,left,height,width))
    for i in random_num_2:
      top = int(i)//int(32)*8
      left = int(i)%int(32)*8
      bbox_3.append((top,left,height,width))
    for i in random_num_3:
      top = int(i)//int(32)*8
      left = int(i)%int(32)*8
      bbox_4.append((top,left,height,width))
    bbox=[bbox_1,bbox_2,bbox_3,bbox_4]
    return bbox
  

  
def patch_bbox_stripe(type_stripe ='vertical',size=8): 
    bbox = []
    bbox_1=[]
    bbox_2=[]
    if type_stripe =='horizontal':
        width =256
        height = size
        num = 256//size//2
        for i in range(num):
            top = i *height*2
            top_1 = i *height*2+height
            left = 0
            bbox_1.append([top,left,height,width])
            bbox_2.append([top_1,left,height,width])
            
    if type_stripe =='vertical':
        width =size
        height = 256
        num = 256//size//2
        for i in range(num):
            top = 0
            left_1 = i *width*2+width
            left = i *width*2
            bbox_1.append([top,left,height,width])
            bbox_2.append([top,left_1,height,width])
    bbox = [bbox_1,bbox_2]
    return bbox
  
  
  
   
   
def random_bbox(img_shape=(256,256), max_bbox_shape=(128, 128), max_bbox_delta=40, min_margin=20):
    """Generate a random bbox for the mask on a given image.

    In our implementation, the max value cannot be obtained since we use
    `np.random.randint`. And this may be different with other standard scripts
    in the community.

    Args:
        img_shape (tuple[int]): The size of a image, in the form of (h, w).
        max_bbox_shape (int | tuple[int]): Maximum shape of the mask box,
            in the form of (h, w). If it is an integer, the mask box will be
            square.
        max_bbox_delta (int | tuple[int]): Maximum delta of the mask box,
            in the form of (delta_h, delta_w). If it is an integer, delta_h
            and delta_w will be the same. Mask shape will be randomly sampled
            from the range of `max_bbox_shape - max_bbox_delta` and
            `max_bbox_shape`. Default: (40, 40).
        min_margin (int | tuple[int]): The minimum margin size from the
            edges of mask box to the image boarder, in the form of
            (margin_h, margin_w). If it is an integer, margin_h and margin_w
            will be the same. Default: (20, 20).

    Returns:
        tuple[int]: The generated box, (top, left, h, w).
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)
        
    img_h, img_w = img_shape[:2]
    max_mask_h, max_mask_w = max_bbox_shape
    max_delta_h, max_delta_w = max_bbox_delta
    margin_h, margin_w = min_margin

    if max_mask_h > img_h or max_mask_w > img_w:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than '
                         f'image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than'
                         f'mask shape {max_bbox_shape}')
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w:
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img'
                         f'shape {img_shape} and mask shape {max_bbox_shape}')

    # get the max value of (top, left)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    # randomly select a (top, left)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    return (top, left, h, w)


def bbox2mask_random(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]
    mask = np.zeros((height, width, 1), dtype=dtype)
    mask_1 = np.zeros((height, width, 1), dtype=dtype)
    mask_2 = np.zeros((height, width, 1), dtype=dtype)
    mask_3 = np.zeros((height, width, 1), dtype=dtype)
    bbox_1 = bbox[0]
    bbox_2 = bbox[1]
    bbox_3 = bbox[2]
    bbox_4 = bbox[3]
    #mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] +bbox[3], :] = 1
    for k in bbox_1:
      mask[k[0]:k[0] + k[2], k[1]:k[1] +k[3],:]=1
    for k in bbox_2:
      mask_1[k[0]:k[0] + k[2], k[1]:k[1] +k[3],:]=1
    for k in bbox_3:
      mask_2[k[0]:k[0] + k[2], k[1]:k[1] +k[3],:]=1
    for k in bbox_4:
      mask_3[k[0]:k[0] + k[2], k[1]:k[1] +k[3],:]=1
    mask_4 = np.ones((height, width, 1), dtype=dtype)
    return mask,mask_1,mask_2,mask_3,mask_4
    
    
    
def bbox2mask_stripe(img_shape, bbox, dtype='uint8'):
    height, width = img_shape[:2]
    mask = np.zeros((height, width, 1), dtype=dtype)
    mask_1 = np.zeros((height, width, 1), dtype=dtype)
    bbox_1 = bbox[0]
    bbox_2 = bbox[1]
    for k in bbox_1:
      mask[k[0]:k[0] + k[2], k[1]:k[1] +k[3],:]=1
    for k in bbox_2:
      mask_1[k[0]:k[0] + k[2], k[1]:k[1] +k[3],:]=1
    mask_4 = np.ones((height, width, 1), dtype=dtype)
    return mask,mask_1,mask_4
    
    
    
def brush_stroke_mask(img_shape,
                      num_vertices=(4, 12),
                      mean_angle=2 * math.pi / 5,
                      angle_range=2 * math.pi / 15,
                      brush_width=(12, 40),
                      max_loops=4,
                      dtype='uint8'):
    """Generate free-form mask.

    The method of generating free-form mask is in the following paper:
    Free-Form Image Inpainting with Gated Convolution.

    When you set the config of this type of mask. You may note the usage of
    `np.random.randint` and the range of `np.random.randint` is [left, right).

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 12).
        mean_angle (float): Mean value of the angle in each vertex. The angle
            is measured in radians. Default: 2 * math.pi / 5.
        angle_range (float): Range of the random angle.
            Default: 2 * math.pi / 15.
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (12, 40).
        max_loops (int): The max number of for loops of drawing strokes.
        dtype (str): Indicate the data type of returned masks.
            Default: 'uint8'.

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    img_h, img_w = img_shape[:2]
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, tuple):
        min_width, max_width = brush_width
    elif isinstance(brush_width, int):
        min_width, max_width = brush_width, brush_width + 1
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    average_radius = math.sqrt(img_h * img_h + img_w * img_w) / 8
    mask = Image.new('L', (img_w, img_h), 0)

    loop_num = np.random.randint(1, max_loops)
    num_vertex_list = np.random.randint(
        min_num_vertices, max_num_vertices, size=loop_num)
    angle_min_list = np.random.uniform(0, angle_range, size=loop_num)
    angle_max_list = np.random.uniform(0, angle_range, size=loop_num)

    for loop_n in range(loop_num):
        num_vertex = num_vertex_list[loop_n]
        angle_min = mean_angle - angle_min_list[loop_n]
        angle_max = mean_angle + angle_max_list[loop_n]
        angles = []
        vertex = []

        # set random angle on each vertex
        angles = np.random.uniform(angle_min, angle_max, size=num_vertex)
        reverse_mask = (np.arange(num_vertex, dtype=np.float32) % 2) == 0
        angles[reverse_mask] = 2 * math.pi - angles[reverse_mask]

        h, w = mask.size

        # set random vertices
        vertex.append((np.random.randint(0, w), np.random.randint(0, h)))
        r_list = np.random.normal(
            loc=average_radius, scale=average_radius // 2, size=num_vertex)
        for i in range(num_vertex):
            r = np.clip(r_list[i], 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        # draw brush strokes according to the vertex and angle list
        draw = ImageDraw.Draw(mask)
        width = np.random.randint(min_width, max_width)
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2,
                          v[0] + width // 2, v[1] + width // 2),
                         fill=1)
    # randomly flip the mask
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.array(mask).astype(dtype=getattr(np, dtype))
    mask = mask[:, :, None]
    return mask


def random_irregular_mask(img_shape,
                          num_vertices=(4, 8),
                          max_angle=4,
                          length_range=(10, 100),
                          brush_width=(10, 40),
                          dtype='uint8'):
    """Generate random irregular masks.

    This is a modified version of free-form mask implemented in
    'brush_stroke_mask'.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    TODO: Rewrite the implementation of this function.

    Args:
        img_shape (tuple[int]): Size of the image.
        num_vertices (int | tuple[int]): Min and max number of vertices. If
            only give an integer, we will fix the number of vertices.
            Default: (4, 8).
        max_angle (float): Max value of angle at each vertex. Default 4.0.
        length_range (int | tuple[int]): (min_length, max_length). If only give
            an integer, we will fix the length of brush. Default: (10, 100).
        brush_width (int | tuple[int]): (min_width, max_width). If only give
            an integer, we will fix the width of brush. Default: (10, 40).
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    h, w = img_shape[:2]

    mask = np.zeros((h, w), dtype=dtype)
    if isinstance(length_range, int):
        min_length, max_length = length_range, length_range + 1
    elif isinstance(length_range, tuple):
        min_length, max_length = length_range
    else:
        raise TypeError('The type of length_range should be int'
                        f'or tuple[int], but got type: {length_range}')
    if isinstance(num_vertices, int):
        min_num_vertices, max_num_vertices = num_vertices, num_vertices + 1
    elif isinstance(num_vertices, tuple):
        min_num_vertices, max_num_vertices = num_vertices
    else:
        raise TypeError('The type of num_vertices should be int'
                        f'or tuple[int], but got type: {num_vertices}')

    if isinstance(brush_width, int):
        min_brush_width, max_brush_width = brush_width, brush_width + 1
    elif isinstance(brush_width, tuple):
        min_brush_width, max_brush_width = brush_width
    else:
        raise TypeError('The type of brush_width should be int'
                        f'or tuple[int], but got type: {brush_width}')

    num_v = np.random.randint(min_num_vertices, max_num_vertices)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        # from the start point, randomly setlect n \in [1, 6] directions.
        direction_num = np.random.randint(1, 6)
        angle_list = np.random.randint(0, max_angle, size=direction_num)
        length_list = np.random.randint(
            min_length, max_length, size=direction_num)
        brush_width_list = np.random.randint(
            min_brush_width, max_brush_width, size=direction_num)
        for direct_n in range(direction_num):
            angle = 0.01 + angle_list[direct_n]
            if i % 2 == 0:
                angle = 2 * math.pi - angle
            length = length_list[direct_n]
            brush_w = brush_width_list[direct_n]
            # compute end point according to the random angle
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1, brush_w)
            start_x, start_y = end_x, end_y
    mask = np.expand_dims(mask, axis=2)

    return mask


def get_irregular_mask(img_shape, area_ratio_range=(0.15, 0.5), **kwargs):
    """Get irregular mask with the constraints in mask ratio

    Args:
        img_shape (tuple[int]): Size of the image.
        area_ratio_range (tuple(float)): Contain the minimum and maximum area
        ratio. Default: (0.15, 0.5).

    Returns:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    mask = random_irregular_mask(img_shape, **kwargs)
    min_ratio, max_ratio = area_ratio_range

    while not min_ratio < (np.sum(mask) /
                           (img_shape[0] * img_shape[1])) < max_ratio:
        mask = random_irregular_mask(img_shape, **kwargs)

    return mask
