#!/bin/sh

rm -f *.joblib

python3 ./train.py ${@}
