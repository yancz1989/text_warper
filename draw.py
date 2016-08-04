# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-08-03 08:15:37
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-08-03 09:16:38

from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import PIL
from PIL import ImageDraw, Image, ImageFont
import scipy as sp
import os
import os.path
import sys
import h5py as h5
import cv2

l = 401
sepr = 0.1
    
def triangle(sz):
  img = Image.new('RGBA', (l, l), (0, 0, 0, 0))
  sep = int(l * sepr)
  h = l - 2 * sep
  bh = h / np.sqrt(3.0)
  pts = np.array([[l / 2, sep], [l / 2 + bh, l - sep], [l / 2 - bh, l - sep]]).astype(np.int)
  center = np.sum(pts, axis = 0) / 3
  pts = [[pts[0, :], pts[1, :], center[:]],
         [pts[1, :], pts[2, :], center[:]],
         [pts[0, :], pts[2, :], center[:]]]
  draw = ImageDraw.Draw(img)
  draw.polygon([(p[0], p[1]) for p in pts[0]], fill = (0, 0, 0, 255), outline = (0, 0, 0, 255))
  draw.polygon([(p[0], p[1]) for p in pts[1]], fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  draw.polygon([(p[0], p[1]) for p in pts[2]], fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  return img.resize((sz, sz), PIL.Image.BILINEAR)
    

def square(sz):
  img = Image.new('RGBA', (l, l), (0, 0, 0, 0))
  sep = int(l * sepr)
  h = l - 2 * sep
  bh = h / np.sqrt(3.0)
  pts = np.array([[sep, sep], [sep, l - sep], [l - sep, l - sep], [l - sep, sep]])
  ptt = np.zeros((5, 2))
  for i in range(4):
    ptt[i, :] = (pts[i, :] + pts[(i + 1) % 4, :]) / 2
  ptt[4, :] = np.sum(pts, axis = 0) / 4
  pt1 = [pts[0, :], ptt[0, :], ptt[4, :], ptt[3, :]]
  pt2 = [pts[1, :], ptt[0, :], ptt[4, :], ptt[1, :]]
  pt3 = [pts[2, :], ptt[2, :], ptt[4, :], ptt[1, :]]
  pt4 = [pts[3, :], ptt[2, :], ptt[4, :], ptt[3, :]]
  draw = ImageDraw.Draw(img)
  draw.polygon([(p[0], p[1]) for p in pt1], fill = (0, 0, 0, 255), outline = (0, 0, 0, 255))
  draw.polygon([(p[0], p[1]) for p in pt2], fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  draw.polygon([(p[0], p[1]) for p in pt3], fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  draw.polygon([(p[0], p[1]) for p in pt4], fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  return img.resize((sz, sz), PIL.Image.BILINEAR)
    
def circle(sz):
  img = Image.new('RGBA', (l, l), (0, 0, 0, 0))
  sep = int(l * sepr)
  h = l - 2 * sep
  bh = h / np.sqrt(3.0)
  draw = ImageDraw.Draw(img)
  draw.pieslice([(sep, sep), (l - sep, l - sep)], 180, 270, 
                fill = (0, 0, 0, 255), outline = (0, 0, 0, 255))
  draw.pieslice([(sep, sep), (l - sep, l - sep)], 0, 90,
                fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  draw.pieslice([(sep, sep), (l - sep, l - sep)], 90, 180,
                fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  draw.pieslice([(sep, sep), (l - sep, l - sep)], 270, 360,
                fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  return img.resize((sz, sz), PIL.Image.BILINEAR)

def square_cross(sz):
  img = Image.new('RGBA', (l, l), (0, 0, 0, 0))
  sep = int(l * sepr)
  h = l - 2 * sep
  bh = h / np.sqrt(3.0)
  pts = np.array([[sep, sep], [sep, l - sep], [l - sep, l - sep], [l - sep, sep]])
  ctr = np.sum(pts, axis = 0) / 4
  pt1 = [pts[0, :], pts[1, :], ctr[:]]
  pt2 = [pts[1, :], pts[2, :], ctr[:]]
  pt3 = [pts[2, :], pts[3, :], ctr[:]]
  pt4 = [pts[3, :], pts[0, :], ctr[:]]
  draw = ImageDraw.Draw(img)
  draw.polygon([(p[0], p[1]) for p in pt1], fill = (0, 0, 0, 255), outline = (0, 0, 0, 255))
  draw.polygon([(p[0], p[1]) for p in pt2], fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  draw.polygon([(p[0], p[1]) for p in pt3], fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  draw.polygon([(p[0], p[1]) for p in pt4], fill = (0, 0, 0, 0), outline = (0, 0, 0, 255))
  return img.resize((sz, sz), PIL.Image.BILINEAR)

def draw_background(imgsz):
  mark_size = int(min(imgsz[0] * 0.1, imgsz[1] * 0.1))
  print((imgsz[1] - mark_size, imgsz[0] - mark_size))
  bg = Image.new('RGBA', imgsz)
  bg.paste(circle(mark_size), (0, 0))
  bg.paste(square(mark_size), (imgsz[0] - mark_size, 0))
  bg.paste(triangle(mark_size), (imgsz[0] - mark_size, imgsz[1] - mark_size))
  bg.paste(square_cross(mark_size), (0, imgsz[1] - mark_size))
  return bg

def show_image(img, figsize = None):
  fig = plt.figure(figsize=figsize)
  plt.imshow(img)

# draw text in box paramter [l, r, u, d],
# align, 0 left, 1 middle, 2 right,
# fsize, font size
# hline, line height = fsize + sep
# text, text wanted
def draw_text(bg, gt, box, align, fnt, fsize, hline, txts):
  '''
  para for align:
    0 for left
    1 for middle
    2 for right
  '''
  width = box[1] - box[0]
  height = box[3] - box[2]
  cols = np.int(np.floor(width / fsize))
  rows = np.int(np.ceil(height / hline))
  draw = ImageDraw.Draw(bg)
  gdraw = ImageDraw.Draw(gt)
  if len(txts) > rows:
    raise Exception('Too many rows...')
  else:
    # box = [left, right, up, down]
    txt_area = [1024, 0, 1024, 0]
    i = 0
    for txt in txts:
      rline = np.int(np.ceil(len(txt) / cols))
      for k in range(rline):
        st = int(cols * k)
        ed = st + cols if st + cols < len(txt) else len(txt)
        ctx = txt[st:ed]
        ltxt = fnt.getsize(ctx)[0]
        if align == 0:
          pos = [box[0], (i + k) * hline + box[2], ltxt + box[0], (i + k) * hline + box[2] + fsize]
        elif align == 1:
          pos = [np.int(box[0] + (width - ltxt) / 2), (i + k) * hline + box[2],
                 np.int(box[1] - (width - ltxt) / 2), (i + k) * hline + box[2] + fsize]
        else:
          pos = [box[1] - ltxt, (i + k) * hline + box[2], box[1], (i + k) * hline + box[2] + fsize]
        if pos[0] < box[0]:
          txt_area[0] = pos[0]
        if pos[1] > box[1]:
          txt_area[1] = pos[1]
        if pos[2] < box[2]:
          txt_area[2] = pos[2]
        if pos[3] < box[3]:
          txt_area[3] = pos[3]
        draw.text((pos[0], pos[1]), ctx, font = fnt, fill = (255, 255, 255))
        gdraw.rectangle((pos[0], pos[1], pos[2], pos[3]), fill = (255, 255, 255))
      i = i + rline
        
  return (np.array([[txt_area[0], txt_area[2], 1.0], [txt_area[1], txt_area[3], 1.0]], dtype = 'float32').T)

def draw_image(bg, img, box = None):
  if box == None:
    box = [0, 0, img.size[0], img.size[1]]
  bw = box[2] - box[0]
  bh = box[3] - box[1]
  _img = img.resize((bw, bh), Image.ANTIALIAS)
  show_image(_img)
  bg.paste(_img, box = box)
  return bg

def draw_img_with_caption(bg, img, caption, box):
  return bg

def draw_text_with_figure(bg, img, text, box):
  return bg