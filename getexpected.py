#!/usr/bin/python -B

import os
import re
import subprocess
import sys

ROOT_PATH = '/course/cs4500f13/Assignments/A5new'
README_PATH1 = '/course/cs4500f13/Assignments/A5new/README'
README_PATH2 = '/course/cs4500f13/Assignments/A5new/D0/readme.wav'
FILE_NAME_MATCHER = re.compile(r'(\w+?\.(wav|mp3))')


def get_expected_from_file(pathname):
  """Gets the expected matches given a file listing them

  Args:
    pathname: The path name to a README file.

  Returns:
    A set of 2-tuples where the elements are paths of matching files.
  """
  derivations = set() # (derived, original)
  with open(pathname, 'r') as f:
    for line in f:
      matches = FILE_NAME_MATCHER.findall(line)
      if len(matches) >= 2:
        derivations.add((matches[0][0].strip(), matches[1][0].strip()))
  # Exceptions to the pattern that have funny names
  excepted_matches = [['z1.ogg', 'y1.wav'],
                      ['z2.txt', 'y5.wav'],
                      ['z4', 'y10.wav'],
                      ['z101', 'z01.wav'],
                      ['z111', 'z11.mp3']]
  for matches in excepted_matches:
    derivations.add((matches[0], matches[1]))
  # Transitivity
  while True:
    addition = set((a, y) for a, b in derivations for x, y in derivations
                   if b == x)
    if addition <= derivations:
      break
    derivations = derivations | addition
  # Commutativity
  for derivation in derivations.copy():
    derivations.add((derivation[1], derivation[0]))
    derivations.add((derivation[0], derivation[0]))
    derivations.add((derivation[1], derivation[1]))
  return derivations


def get_full_path(search_dir, name):
  """Searches for the file with the given name within the given
  directory.

  Args:
    search_dir: A path to a directory.
    name: A file name.

  Returns:
    The full path of the file, if found; otherwise, None.
  """
  return (subprocess.Popen(['find', search_dir, '-name', name, '-print',
                            '-quit'],
                           stdout=subprocess.PIPE)
          .communicate()[0]).rstrip()


def main():
  derivations = get_expected_from_file(README_PATH1)
  derivations = derivations | get_expected_from_file(README_PATH2)
  dirs = set()
  paths = {}
  for name1, name2 in derivations:
    path1 = get_full_path(ROOT_PATH, name1)
    path2 = get_full_path(ROOT_PATH, name2)
    if path1:
      dirs.add(os.path.dirname(path1))
      paths[name1] = path1
    if path2:
      dirs.add(os.path.dirname(path2))
      paths[name2] = path2
  tests = 0
  tests_failed = 0
  total_seconds = None
  for name1 in paths:
    for name2 in paths:
      path1 = paths[name1]
      path2 = paths[name2]
      out, time = (subprocess.Popen(['time', './p4500', '-f', path1, '-f',
                                     path2],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
                   .communicate())
      out = out.rstrip()
      try:
        time = time.split()[2]
      except IndexError:
        time = '???'
      if time.endswith('elapsed'):
        time = time[:-len('elapsed')]
      tests = tests + 1
      print('{0}\t{1}\t{2}'.format(name1, name2, time))
      try:
        minutes, seconds = re.match(r'(\d+):([\d.]+)', time).groups()
        total_seconds = int(minutes) * 60 + float(seconds)
      except (AttributeError, ValueError):
        total_seconds = None
      did_match = out.startswith('MATCH')
      should_match = (name1, name2) in derivations
      if did_match is not should_match:
        print('FAIL {0} {1} should {2}MATCH'
              .format(name1, name2, '' if should_match else 'NOT '))
        tests_failed = tests_failed + 1
  print('FAILED: {0} / {1} ({2}%)'
        .format(tests_failed, tests, tests_failed * 100 // tests))
  print('TIME: {0}:{1:.2f}'.format(int(total_seconds // 60),
      total_seconds % 60))


if __name__ == '__main__':
  main()
