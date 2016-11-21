# -*- coding: utf-8 -*-
# @Author: yancz1989
# @Date:   2016-10-19 19:56:04
# @Last Modified by:   yancz1989
# @Last Modified time: 2016-11-09 09:52:36

import sys
import numpy as np

def parse(f):
  stats = []
  stat = {}
  stat['train'] = []
  stat['val'] = []
  stat['epoches'] = []
  epochIdx = []
  with open(f) as f:
    lines = f.readlines()
    for line in lines:
      if line[0] == 'c':
        if len(stat) != 0:
          stats.append(stat)
        stat['train'] = []
        stat['val'] = []
        stat['epoches'] = []
      elif line[0] == 'e':
        words = line.split(' ')
        stat['epoches'].append([int(words[1])] + [float(words[i]) for i in epochIdx[1:]])
      elif line[0] == 't' or line[0] == 'v':
        words = line.split(' ')
        stat[words[0]].append([int(words[1])] + [float(words[i]) for i in range(2, len(words) - 1)])
        if epochIdx == []:
          ncomps = len(words) - 2
          epochIdx = [1, 3] + range(6, 6 + ncomps) + range(6 + ncomps + 1, 6 + 2 * ncomps + 1)
    stats.append(stat)
  return stats

def show(x, y):
  print('log of experiment %d:%d' % (x, y))
  epoches = parse('config/exp' + str(x) + '/' + str(y) + '.log')[0]['epoches']
  log = np.array([epoch for epoch in epoches])
  return epoches
  
if __name__ == '__main__':
  epoches = show(int(sys.argv[1]), int(sys.argv[2]))
  for epoch in epoches:
    print(epoch)