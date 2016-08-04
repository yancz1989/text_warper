# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-04-12 11:06:29
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-04-13 16:18:48
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2015-10-11 21:30:19
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-02-29 16:16:39

import numpy as np
import scipy as sp
from PIL import Image,ImageDraw,ImageFont
from numpy import random as rnd
from scipy import io as sio
from util import *
import cv2

def namegen(sex, boyn, girln, given):
    nid = rnd.randint(0, len(boyn if sex == 0 else girln) >> 1) << 1
    return ''.join([given[rnd.randint(0, len(given))], 
        boyn[nid : nid + 2] if sex == 0 else girln[nid : nid + 2]])

def draw_text(img, pos_c, cont_font, name, addr, alpha, sex, prov, lp):
    draw = ImageDraw.Draw(img)  
    draw.text((pos_c[0][0], pos_c[0][1]), name, (0, 0, 0),
        font = cont_font)
    draw.text((pos_c[1][0], pos_c[1][1]), sex, (0, 0, 0), font = cont_font)
    draw.text((pos_c[2][0], pos_c[2][1]), ''.join([str(t) for t in rnd.randint(
        0, 10, size=(rnd.randint(7, 11)))]), (0, 0, 0), font = cont_font)
    draw.text((pos_c[3][0], pos_c[3][1]), addr[0] + addr[1], (0, 0, 0), font = cont_font)
    draw.text((pos_c[4][0], pos_c[4][1]), ''.join([str(rnd.randint(1965, 2005)), u'年',
         str(rnd.randint(1, 13)), u'月']), (0, 0, 0), font = cont_font)
    draw.text((pos_c[5][0], pos_c[5][1]), prov[rnd.randint(0,len(prov))],
        (0,0,0), font = cont_font)
    draw.text((pos_c[6][0], pos_c[6][1]), ''.join(['#' + str(rnd.randint(1, 20))]), (0, 0, 0),
        font = cont_font)
    return img

def imggen(draw_func, root, ids, m, n, r, boyn, girln, given, addr, alpha, sex, label_size,
        cont_size, prov, pos_l, left, right, pos_c, label, label_font, cont_font):
    for lp in ids:
        sexi = rnd.randint(0, 2)
        img = draw_func(Image.open(root + 'template.bmp'), pos_c, cont_font, 
            namegen(sexi, boyn, girln, given), [addr[(lp << 1) + 1], addr[lp << 1]],
            alpha, sex[sexi], prov, lp - 1)
        img.save(root + 'clean/' + str(lp) + '.bmp', 'bmp')
    
def img_generate(root, ids = None, cnt = 800):
    left = [[50, 81], [206, 128]]
    right = [[120, 101], [300, 148]]
    r = 60.0 / 637
    m = 60
    n = int(398 * r)
    names = get_string_list(root + 'meta/names.txt')
    addrs = get_string_list(root + 'meta/exaddr.txt')
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    sex = u'男女'
    label_size = 24
    cont_size = 27
    prov = [u'河南',u'河北',u'山东',u'江苏',u'浙江',u'新疆',u'辽宁',u'湖北',
        u'四川',u'西藏',u'广东',u'安徽', u'北京', u'上海', u'黑龙江', u'吉林',
        u'青海', u'内蒙古', u'江西', u'安徽', u'湖南', u'福建', u'广西',
        u'云南', u'贵州', u'陕西', u'甘肃', u'宁夏']
    pos_l = [[75, 52],[75, 122],[75, 192], [75, 262], [310, 52], [310, 122], [310, 192]]
    pos_c = [[145, 58],[145, 126],[145, 198], [195, 266], [420, 58], [370, 126], [400, 198]]
    label = [u'姓名', u'性别', u'学号', u'通讯地址', u'出生日期', u'籍贯', u'宿舍楼']
    cont_font = ImageFont.truetype(root + 'fonts/Songti.ttc', 18)
    label_font = ImageFont.truetype(root + 'fonts/Songti.ttc', 24)
    given = get_str_from_file(root + 'meta/given.txt')
    boyn = get_str_from_file(root + 'meta/boy.txt')
    girln = get_str_from_file(root + 'meta/girl.txt')
    addr = get_string_list(root + 'meta/address.txt')
    imggen(draw_text, root, range(1, cnt + 1), m, n, r, boyn, girln, given, addr, 
        alpha, sex, label_size, cont_size, prov, pos_l, left, right, pos_c, label,
         label_font, cont_font)

if __name__ == '__main__':
    root = '/home/yancz/work/dl_warp/data/'
    ret = img_generate(root, None)


