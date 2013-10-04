#!/usr/bin/python

import sys

def check_args():
  return len(sys.argv) == 3


if not check_args():
  print 'ERROR Usage: ./p4500 <path> <path>'


sys.exit(0)

