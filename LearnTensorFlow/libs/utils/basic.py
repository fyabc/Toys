#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
from datetime import datetime
import math

__author__ = 'fyabc'


def time_tensorflow_run(session, target, info_string, num_batches):
    n_steps_burn_in = 10    # Skip several first steps
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + n_steps_burn_in):
        start_time = time.time()
        session.run(target)
        duration = time.time() - start_time

        if i >= n_steps_burn_in:
            if not i % 10:
                print('{:s}: step {}, duration = {:.3f}s'.format(datetime.now(), i - n_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration ** 2

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn ** 2
    sd = math.sqrt(vr)
    print('{}: {} across {} steps, {:.3f} +/- {:.3f} sec / batch'.format(
        datetime.now(), info_string, num_batches, mn, sd))
