#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 10:31:37 2017

@author: joe
"""

import uinput
import time
time.sleep(10)
device = uinput.Device([
        uinput.KEY_W,
        uinput.KEY_A,
        uinput.KEY_S,
        uinput.KEY_D,
        ])

device.emit_click(uinput.KEY_W)
device.emit_click(uinput.KEY_A)
device.emit_click(uinput.KEY_S)
device.emit_click(uinput.KEY_D)
