#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:33:10 2022

@author: hemerson
"""

# Hide PyGame Welcome
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Import packages
from .collection import collect_sample
from .evaluation import test_policy
from .processing import clean_sample, get_batch
