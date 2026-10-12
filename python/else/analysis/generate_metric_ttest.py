"""
統計分析腳本：
1. Table 1: Sequence Effect (Group 1 vs Group 2 組間檢定 - Mann-Whitney U & Independent t-test)
2. Table 2: Paired Samples Tests (LE: S2 vs S1; AE: Adaptive vs Static - Paired t-test & Wilcoxon)
3. Table 2b: One-Sample Tests on Deltas against 0 (LE Δ vs 0; AE Δ vs 0 - One-sample t-test & Wilcoxon)
4. Table 3: Linear Mixed Model (LMM) & Two-way Mixed ANOVA
"""

import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

warnings.filterwarnings('ignore')

raw_data_13 = {
    35: {
        's1': {
            'cond': 'B',
            'on': [0.525, 0.525, 0.6, 0.65, 0.6, 0.6, 0.575],
            'off_run': [0.606, 0.597, 0.602, 0.659, 0.606, 0.605, 0.652],
            'off_sess': 0.559,
            'Spectral Saliency': [
                28.64,
                39.03,
                32.36,
                37.11,
                21.53,
                23.80,
                18.87,
            ],
            'Spatial Saliency': [
                38.39,
                30.06,
                42.14,
                43.19,
                39.66,
                41.63,
                18.11,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.55, 0.5, 0.525, 0.55, 0.575, 0.525, 0.5],
            'off_run': [0.547, 0.547, 0.62, 0.588, 0.573, 0.648, 0.648],
            'off_sess': 0.582,
            'Spectral Saliency': [
                15.28,
                30.54,
                4.23,
                40.70,
                18.88,
                22.71,
                59.61,
            ],
            'Spatial Saliency': [
                37.13,
                48.17,
                62.53,
                26.55,
                29.21,
                43.57,
                33.80,
            ],
        },
    },
    37: {
        's1': {
            'cond': 'A',
            'on': [0.45, 0.5, 0.6, 0.6, 0.625, 0.95, 0.897],
            'off_run': [0.613, 0.614, 0.83, 0.731, 0.7, 0.994, 0.998],
            'off_sess': 0.735,
            'Spectral Saliency': [
                25.97,
                15.76,
                37.83,
                55.35,
                29.42,
                78.49,
                79.90,
            ],
            'Spatial Saliency': [
                29.90,
                44.40,
                29.50,
                16.74,
                37.61,
                53.53,
                23.67,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.55, 0.9, 0.975, 0.925, 0.95, 0.975],
            'off_run': [0.855, 0.986, 0.994, 0.995, 1.0, 0.998],
            'off_sess': 0.965,
            'Spectral Saliency': [
                37.90,
                68.53,
                '...',
                75.41,
                70.44,
                76.43,
                80.45,
            ],
            'Spatial Saliency': [
                51.86,
                39.81,
                '...',
                43.70,
                35.95,
                44.14,
                29.08,
            ],
        },
    },
    38: {
        's1': {
            'cond': 'B',
            'on': [0.475, 0.5, 0.5, 0.525, 0.5, 0.575, 0.525],
            'off_run': [0.531, 0.634, 0.658, 0.658, 0.566, 0.697, 0.7],
            'off_sess': 0.583,
            'Spectral Saliency': [
                23.94,
                39.76,
                50.62,
                52.75,
                21.85,
                34.84,
                27.44,
            ],
            'Spatial Saliency': [
                34.16,
                50.55,
                37.72,
                40.16,
                34.74,
                50.61,
                47.02,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.6, 0.5, 0.564, 0.425, 0.525, 0.6, 0.575],
            'off_run': [0.559, 0.606, 0.705, 0.681, 0.734, 0.611, 0.567],
            'off_sess': 0.575,
            'Spectral Saliency': [
                32.48,
                39.71,
                52.45,
                30.27,
                53.29,
                79.14,
                24.21,
            ],
            'Spatial Saliency': [
                19.24,
                33.99,
                57.73,
                40.59,
                33.07,
                57.87,
                32.47,
            ],
        },
    },
    40: {
        's1': {
            'cond': 'B',
            'on': [0.575, 0.525, 0.5, 0.525, 0.6, 0.5, 0.525],
            'off_run': [0.647, 0.655, 0.589, 0.602, 0.58, 0.591, 0.594],
            'off_sess': 0.542,
            'Spectral Saliency': [
                28.13,
                31.71,
                23.72,
                37.66,
                16.40,
                19.65,
                17.69,
            ],
            'Spatial Saliency': [
                38.91,
                48.65,
                36.28,
                36.13,
                29.33,
                32.14,
                32.68,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.55, 0.55, 0.475, 0.55, 0.5, 0.525, 0.59],
            'off_run': [0.58, 0.597, 0.602, 0.569, 0.562, 0.631, 0.634],
            'off_sess': 0.569,
            'Spectral Saliency': [
                21.06,
                30.49,
                35.60,
                10.70,
                22.73,
                30.27,
                45.96,
            ],
            'Spatial Saliency': [
                32.16,
                29.64,
                30.85,
                46.63,
                27.87,
                46.88,
                32.20,
            ],
        },
    },
    41: {
        's1': {
            'cond': 'B',
            'on': [0.45, 0.575, 0.6, 0.8, 0.675, 0.7, 0.625],
            'off_run': [0.6, 0.686, 0.636, 0.731, 0.728, 0.767, 0.67],
            'off_sess': 0.682,
            'Spectral Saliency': [
                18.12,
                45.01,
                24.69,
                84.04,
                73.99,
                73.67,
                25.57,
            ],
            'Spatial Saliency': [
                31.01,
                41.15,
                32.24,
                45.07,
                51.02,
                50.85,
                32.77,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.6, 0.725, 0.725, 0.65, 0.775, 0.575, 0.625],
            'off_run': [0.681, 0.614, 0.686, 0.697, 0.686, 0.623, 0.628],
            'off_sess': 0.639,
            'Spectral Saliency': [
                26.39,
                21.25,
                77.83,
                82.15,
                30.42,
                17.29,
                59.44,
            ],
            'Spatial Saliency': [
                24.34,
                29.40,
                70.30,
                67.28,
                57.00,
                35.02,
                53.54,
            ],
        },
    },
    42: {
        's1': {
            'cond': 'A',
            'on': [0.6, 0.55, 0.5, 0.425, 0.5, 0.55, 0.525],
            'off_run': [0.577, 0.609, 0.583, 0.584, 0.608, 0.592, 0.569],
            'off_sess': 0.559,
            'Spectral Saliency': [
                9.45,
                12.97,
                24.96,
                9.94,
                33.80,
                48.03,
                10.79,
            ],
            'Spatial Saliency': [
                33.21,
                40.14,
                34.99,
                40.58,
                38.85,
                29.35,
                36.99,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.575, 0.55, 0.5, 0.525, 0.625, 0.5, 0.5],
            'off_run': [0.598, 0.556, 0.636, 0.573, 0.614, 0.645, 0.609],
            'off_sess': 0.542,
            'Spectral Saliency': [
                30.40,
                25.96,
                41.24,
                28.77,
                29.41,
                26.42,
                41.95,
            ],
            'Spatial Saliency': [
                51.95,
                22.67,
                31.54,
                37.35,
                22.74,
                32.97,
                38.97,
            ],
        },
    },
    43: {
        's1': {
            'cond': 'A',
            'on': [0.5, 0.5, 0.525, 0.425, 0.525, 0.625, 0.475],
            'off_run': [0.569, 0.575, 0.617, 0.614, 0.661, 0.752, 0.733],
            'off_sess': 0.661,
            'Spectral Saliency': [
                18.74,
                15.01,
                65.75,
                13.59,
                16.80,
                63.97,
                82.18,
            ],
            'Spatial Saliency': [
                22.43,
                36.45,
                78.27,
                25.71,
                47.46,
                39.01,
                40.44,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.45, 0.675, 0.525, 0.55, 0.675, 0.65, 0.625],
            'off_run': [0.656, 0.636, 0.617, 0.697, 0.611, 0.725, 0.683],
            'off_sess': 0.658,
            'Spectral Saliency': [
                47.48,
                65.82,
                46.74,
                46.34,
                45.92,
                62.68,
                72.77,
            ],
            'Spatial Saliency': [
                51.38,
                54.29,
                39.27,
                45.77,
                35.25,
                51.29,
                37.49,
            ],
        },
    },
    44: {
        's1': {
            'cond': 'B',
            'on': [0.675, 0.725, 0.8, 0.975, 0.975, 0.9, 0.875],
            'off_run': [0.766, 0.869, 0.784, 0.841, 0.855, 0.83, 0.833],
            'off_sess': 0.847,
            'Spectral Saliency': [
                73.05,
                83.39,
                79.59,
                78.25,
                74.77,
                77.86,
                83.02,
            ],
            'Spatial Saliency': [
                62.59,
                58.23,
                43.20,
                55.13,
                60.80,
                57.36,
                57.58,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.475, 0.825, 0.925, 0.875, 0.825, 0.875],
            'off_run': [0.809, 0.87, 0.892, 0.87, 0.928, 0.895],
            'off_sess': 0.874,
            'Spectral Saliency': [
                79.58,
                78.22,
                '...',
                82.10,
                74.38,
                79.04,
                83.23,
            ],
            'Spatial Saliency': [
                52.56,
                56.68,
                '...',
                72.24,
                62.84,
                78.99,
                69.10,
            ],
        },
    },
    45: {
        's1': {
            'cond': 'B',
            'on': [0.425, 0.425, 0.575, 0.525, 0.5, 0.625, 0.45],
            'off_run': [0.597, 0.675, 0.708, 0.658, 0.68, 0.616, 0.72],
            'off_sess': 0.615,
            'Spectral Saliency': [
                43.24,
                51.22,
                50.71,
                18.65,
                69.62,
                13.87,
                62.72,
            ],
            'Spatial Saliency': [
                34.21,
                56.77,
                45.68,
                27.42,
                44.17,
                44.92,
                55.21,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.6, 0.8, 0.85, 0.75, 0.925, 0.9],
            'off_run': [0.637, 0.841, 0.947, 0.881, 0.98, 0.978],
            'off_sess': 0.83,
            'Spectral Saliency': [
                31.16,
                66.86,
                '...',
                75.17,
                74.48,
                59.48,
                40.48,
            ],
            'Spatial Saliency': [
                33.58,
                49.55,
                '...',
                51.59,
                35.15,
                41.26,
                34.92,
            ],
        },
    },
    47: {
        's1': {
            'cond': 'B',
            'on': [0.675, 0.65, 0.6, 0.625, 0.5, 0.475, 0.575],
            'off_run': [0.681, 0.614, 0.689, 0.652, 0.619, 0.591, 0.656],
            'off_sess': 0.65,
            'Spectral Saliency': [
                23.34,
                31.12,
                50.75,
                45.05,
                43.70,
                66.89,
                67.88,
            ],
            'Spatial Saliency': [
                41.59,
                45.05,
                45.91,
                47.93,
                50.13,
                47.36,
                31.09,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.675, 0.525, 0.5, 0.575, 0.55, 0.55, 0.525],
            'off_run': [0.598, 0.584, 0.592, 0.577, 0.586, 0.672, 0.647],
            'off_sess': 0.572,
            'Spectral Saliency': [
                13.59,
                36.95,
                53.80,
                65.51,
                35.49,
                52.88,
                15.92,
            ],
            'Spatial Saliency': [
                32.99,
                44.52,
                19.36,
                22.88,
                23.12,
                40.27,
                42.40,
            ],
        },
    },
    48: {
        's1': {
            'cond': 'B',
            'on': [0.525, 0.475, 0.45, 0.45, 0.475, 0.475, 0.575],
            'off_run': [0.656, 0.656, 0.631, 0.637, 0.667, 0.598, 0.561],
            'off_sess': 0.589,
            'Spectral Saliency': [
                46.07,
                19.55,
                13.46,
                43.12,
                56.70,
                33.10,
                14.39,
            ],
            'Spatial Saliency': [
                25.20,
                32.26,
                35.02,
                32.82,
                53.59,
                45.10,
                35.55,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.55, 0.525, 0.5, 0.65, 0.525, 0.5, 0.45],
            'off_run': [0.614, 0.581, 0.595, 0.611, 0.641, 0.589, 0.597],
            'off_sess': 0.535,
            'Spectral Saliency': [
                22.59,
                17.28,
                26.20,
                42.14,
                31.66,
                33.14,
                33.96,
            ],
            'Spatial Saliency': [
                33.79,
                33.79,
                38.43,
                47.80,
                32.53,
                55.47,
                46.74,
            ],
        },
    },
    50: {
        's1': {
            'cond': 'A',
            'on': [0.5, 0.65, 0.5, 0.625, 0.775, 0.65, 0.575],
            'off_run': [0.7, 0.583, 0.636, 0.752, 0.889, 0.798, 0.8],
            'off_sess': 0.715,
            'Spectral Saliency': [
                19.55,
                9.28,
                28.54,
                46.46,
                79.07,
                76.13,
                74.65,
            ],
            'Spatial Saliency': [
                40.51,
                33.02,
                16.16,
                43.18,
                25.55,
                25.96,
                38.18,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.7, 0.7, 0.525, 0.55, 0.6, 0.75, 0.825],
            'off_run': [0.692, 0.759, 0.614, 0.686, 0.811, 0.92, 0.945],
            'off_sess': 0.718,
            'Spectral Saliency': [
                70.10,
                71.21,
                15.98,
                48.24,
                75.58,
                75.92,
                79.53,
            ],
            'Spatial Saliency': [
                29.92,
                38.12,
                31.63,
                41.51,
                30.11,
                41.72,
                17.83,
            ],
        },
    },
    51: {
        's1': {
            'cond': 'B',
            'on': [0.425, 0.35, 0.5, 0.55, 0.45, 0.575, 0.5],
            'off_run': [0.547, 0.684, 0.714, 0.616, 0.655, 0.669, 0.664],
            'off_sess': 0.61,
            'Spectral Saliency': [
                16.18,
                14.02,
                22.20,
                17.28,
                64.88,
                41.47,
                22.01,
            ],
            'Spatial Saliency': [
                36.70,
                73.84,
                44.75,
                36.98,
                39.28,
                35.05,
                30.33,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.775, 0.475, 0.575, 0.625, 0.55, 0.666],
            'off_run': [0.658, 0.68, 0.587, 0.642, 0.594, 0.666],
            'off_sess': 0.613,
            'Spectral Saliency': [
                51.09,
                66.56,
                '...',
                40.86,
                25.46,
                29.50,
                34.83,
            ],
            'Spatial Saliency': [
                47.16,
                59.96,
                59.76,
                37.74,
                22.68,
                58.19,
            ],
        },
    },
    52: {
        's1': {
            'cond': 'B',
            'on': [0.5, 0.475, 0.425, 0.5, 0.5, 0.45, 0.475],
            'off_run': [0.586, 0.645, 0.583, 0.614, 0.608, 0.619, 0.514],
            'off_sess': 0.547,
            'Spectral Saliency': [
                14.20,
                45.59,
                53.14,
                41.57,
                33.81,
                30.93,
                32.80,
            ],
            'Spatial Saliency': [
                31.10,
                37.09,
                15.87,
                61.14,
                32.68,
                28.44,
                32.66,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.575, 0.6, 0.475, 0.575, 0.625, 0.6, 0.45],
            'off_run': [0.559, 0.566, 0.68, 0.609, 0.63, 0.562, 0.609],
            'off_sess': 0.565,
            'Spectral Saliency': [
                47.10,
                22.90,
                30.96,
                27.99,
                43.07,
                47.74,
                40.63,
            ],
            'Spatial Saliency': [
                57.39,
                40.32,
                '...',
                34.77,
                34.68,
                47.73,
                49.44,
            ],
        },
    },
    54: {
        's1': {
            'cond': 'B',
            'on': [0.55, 0.525, 0.525, 0.55, 0.625, 0.55, 0.525],
            'off_run': [0.623, 0.662, 0.63, 0.63, 0.678, 0.681, 0.7],
            'off_sess': 0.632,
            'Spectral Saliency': [
                20.79,
                13.20,
                22.70,
                22.79,
                7.80,
                35.83,
                12.57,
            ],
            'Spatial Saliency': [
                46.32,
                27.46,
                29.67,
                31.03,
                40.35,
                47.22,
                38.73,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.475, 0.75, 0.65, 0.5, 0.7, 0.575],
            'off_run': [0.641, 0.683, 0.731, 0.658, 0.78, 0.811],
            'off_sess': 0.708,
            'Spectral Saliency': [
                22.85,
                68.37,
                '...',
                52.01,
                63.02,
                71.85,
                78.39,
            ],
            'Spatial Saliency': [
                38.99,
                44.08,
                '...',
                50.09,
                26.85,
                43.62,
                33.05,
            ],
        },
    },
    55: {
        's1': {
            'cond': 'A',
            'on': [0.575, 0.5, 0.55, 0.55, 0.65, 0.55, 0.55],
            'off_run': [0.581, 0.583, 0.592, 0.688, 0.614, 0.664, 0.63],
            'off_sess': 0.589,
            'Spectral Saliency': [
                3.52,
                7.50,
                5.05,
                12.01,
                14.43,
                65.19,
                21.69,
            ],
            'Spatial Saliency': [
                37.28,
                36.34,
                26.79,
                57.04,
                36.73,
                46.10,
                42.62,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.45, 0.45, 0.475, 0.475, 0.55, 0.475, 0.5],
            'off_run': [0.519, 0.566, 0.586, 0.602, 0.672, 0.609, 0.577],
            'off_sess': 0.572,
            'Spectral Saliency': [
                23.29,
                14.15,
                24.87,
                16.62,
                35.60,
                4.72,
                15.65,
            ],
            'Spatial Saliency': [
                40.40,
                34.66,
                32.96,
                21.60,
                49.31,
                40.07,
                30.43,
            ],
        },
    },
    57: {
        's1': {
            'cond': 'B',
            'on': [0.525, 0.65, 0.55, 0.5, 0.625, 0.55, 0.525],
            'off_run': [0.572, 0.623, 0.588, 0.595, 0.577, 0.641, 0.623],
            'off_sess': 0.598,
            'Spectral Saliency': [
                11.89,
                15.09,
                13.33,
                25.88,
                26.12,
                18.72,
                13.92,
            ],
            'Spatial Saliency': [
                37.90,
                35.16,
                34.24,
                28.22,
                42.08,
                23.24,
                32.23,
            ],
        },
        's2': {
            'cond': 'A',
            'on': [0.625, 0.5, 0.525, 0.5, 0.525, 0.525, 0.525],
            'off_run': [0.655, 0.58, 0.631, 0.673, 0.588, 0.583, 0.602],
            'off_sess': 0.55,
            'Spectral Saliency': [
                26.74,
                58.78,
                35.27,
                36.79,
                25.47,
                30.59,
                21.24,
            ],
            'Spatial Saliency': [
                52.12,
                38.35,
                38.45,
                53.74,
                40.25,
                30.00,
                40.24,
            ],
        },
    },
    58: {
        's1': {
            'cond': 'A',
            'on': [0.65, 0.475, 0.575, 0.59, 0.725, 0.5, 0.6],
            'off_run': [0.669, 0.689, 0.597, 0.72, 0.795, 0.661, 0.627],
            'off_sess': 0.624,
            'Spectral Saliency': [
                30.65,
                34.80,
                23.45,
                43.63,
                45.45,
                41.52,
                37.35,
            ],
            'Spatial Saliency': [
                28.40,
                41.75,
                29.47,
                57.19,
                63.08,
                38.39,
                38.34,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.6, 0.65, 0.525, 0.725, 0.825, 0.675, 0.8],
            'off_run': [0.75, 0.75, 0.728, 0.794, 0.792, 0.894, 0.936],
            'off_sess': 0.803,
            'Spectral Saliency': [
                63.47,
                45.13,
                51.87,
                73.51,
                42.78,
                50.77,
                73.29,
            ],
            'Spatial Saliency': [
                53.45,
                43.17,
                49.55,
                34.62,
                49.98,
                53.46,
                48.06,
            ],
        },
    },
    63: {
        's1': {
            'cond': 'A',
            'on': [0.625, 0.7, 0.55, 0.525, 0.5, 0.65, 0.675],
            'off_run': [0.592, 0.627, 0.633, 0.73, 0.628, 0.559, 0.664],
            'off_sess': 0.578,
            'Spectral Saliency': [
                26.37,
                74.16,
                50.25,
                70.10,
                21.92,
                43.93,
                33.03,
            ],
            'Spatial Saliency': [
                23.99,
                45.26,
                35.16,
                44.61,
                63.25,
                36.68,
                59.50,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.5, 0.775, 0.725, 0.625, 0.7, 0.7],
            'off_run': [0.608, 0.762, 0.798, 0.914, 0.883, 0.814],
            'off_sess': 0.764,
            'Spectral Saliency': [
                15.83,
                62.16,
                '...',
                71.48,
                80.01,
                84.56,
                72.27,
            ],
            'Spatial Saliency': [
                41.39,
                46.92,
                '...',
                68.88,
                71.73,
                71.42,
                66.49,
            ],
        },
    },
    64: {
        's1': {
            'cond': 'A',
            'on': [0.55, 0.55, 0.5, 0.45, 0.575, 0.615, 0.675],
            'off_run': [0.586, 0.684, 0.731, 0.777, 0.764, 0.728, 0.802],
            'off_sess': 0.724,
            'Spectral Saliency': [
                12.68,
                40.29,
                46.21,
                72.47,
                49.93,
                77.78,
                66.01,
            ],
            'Spatial Saliency': [
                33.31,
                34.85,
                58.87,
                67.28,
                75.51,
                52.77,
                58.86,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.5, 0.675, 0.725, 0.575, 0.75, 0.45, 0.6],
            'off_run': [0.741, 0.731, 0.681, 0.664, 0.702, 0.661, 0.636],
            'off_sess': 0.713,
            'Spectral Saliency': [
                65.14,
                59.39,
                63.47,
                21.63,
                61.59,
                48.40,
                31.36,
            ],
            'Spatial Saliency': [
                69.66,
                49.52,
                54.18,
                32.23,
                68.92,
                60.12,
                33.07,
            ],
        },
    },
    65: {
        's1': {
            'cond': 'A',
            'on': [0.5, 0.625, 0.575, 0.5, 0.5, 0.475, 0.475],
            'off_run': [0.65, 0.616, 0.613, 0.656, 0.58, 0.636, 0.611],
            'off_sess': 0.582,
            'Spectral Saliency': [
                26.23,
                24.70,
                17.45,
                24.83,
                29.03,
                53.08,
                14.63,
            ],
            'Spatial Saliency': [
                46.31,
                33.20,
                40.63,
                40.96,
                40.72,
                67.87,
                38.54,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.575, 0.525, 0.475, 0.525, 0.525, 0.4, 0.475],
            'off_run': [0.689, 0.638, 0.567, 0.636, 0.678, 0.625, 0.6],
            'off_sess': 0.629,
            'Spectral Saliency': [
                56.46,
                57.32,
                32.10,
                55.64,
                62.41,
                49.64,
                45.26,
            ],
            'Spatial Saliency': [
                43.20,
                46.72,
                39.16,
                53.32,
                53.61,
                47.59,
                38.12,
            ],
        },
    },
    68: {
        's1': {
            'cond': 'A',
            'on': [0.55, 0.475, 0.6, 0.675, 0.45, 0.525, 0.55],
            'off_run': [0.561, 0.583, 0.637, 0.661, 0.547, 0.653, 0.694],
            'off_sess': 0.557,
            'Spectral Saliency': [
                20.52,
                26.87,
                52.76,
                56.30,
                27.31,
                50.18,
                24.12,
            ],
            'Spatial Saliency': [
                34.59,
                27.69,
                41.47,
                46.63,
                37.04,
                46.23,
                35.39,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.525, 0.575, 0.6, 0.575, 0.575, 0.45, 0.55],
            'off_run': [0.642, 0.652, 0.616, 0.605, 0.623, 0.589, 0.594],
            'off_sess': 0.597,
            'Spectral Saliency': [
                36.30,
                43.77,
                13.99,
                38.03,
                57.80,
                18.10,
                37.96,
            ],
            'Spatial Saliency': [
                59.81,
                42.08,
                37.94,
                50.59,
                60.57,
                36.48,
                30.40,
            ],
        },
    },
    69: {
        's1': {
            'cond': 'A',
            'on': [0.575, 0.5, 0.775, 0.95, 0.95, 0.9, 1.0],
            'off_run': [0.739, 0.759, 0.837, 0.895, 0.945, 0.891, 0.956],
            'off_sess': 0.748,
            'Spectral Saliency': [
                78.69,
                64.52,
                62.02,
                72.79,
                60.55,
                48.32,
                35.93,
            ],
            'Spatial Saliency': [
                66.37,
                72.16,
                66.47,
                67.08,
                87.38,
                54.84,
                61.58,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.675, 0.85, 0.95, 0.925, 0.95, 0.95],
            'off_run': [0.756, 0.905, 0.969, 0.947, 0.984, 0.952],
            'off_sess': 0.91,
            'Spectral Saliency': [
                28.61,
                40.47,
                '...',
                50.52,
                29.91,
                17.91,
                42.66,
            ],
            'Spatial Saliency': [
                57.06,
                72.43,
                '...',
                87.94,
                56.80,
                74.13,
                70.81,
            ],
        },
    },
    70: {
        's1': {
            'cond': 'A',
            'on': [0.475, 0.55, 0.575, 0.725, 0.625, 0.5, 0.625],
            'off_run': [0.728, 0.622, 0.622, 0.689, 0.616, 0.58, 0.661],
            'off_sess': 0.618,
            'Spectral Saliency': [
                51.23,
                37.45,
                22.64,
                50.13,
                37.37,
                30.42,
                57.00,
            ],
            'Spatial Saliency': [
                55.78,
                40.74,
                38.04,
                58.76,
                46.74,
                31.58,
                42.39,
            ],
        },
        's2': {
            'cond': 'B',
            'on': [0.475, 0.55, 0.775, 0.75, 0.675, 0.675, 0.825],
            'off_run': [0.686, 0.802, 0.822, 0.836, 0.797, 0.822, 0.728],
            'off_sess': 0.831,
            'Spectral Saliency': [
                48.63,
                74.56,
                81.05,
                82.82,
                83.75,
                77.76,
                79.00,
            ],
            'Spatial Saliency': [
                57.65,
                83.13,
                65.51,
                65.60,
                74.16,
                74.55,
                85.02,
            ],
        },
    },
}


# ==========================================
# 統計 Helpers
# ==========================================
def safe_wilcoxon(x, y=None):
  if y is not None:
    diff = np.array(x, dtype=float) - np.array(y, dtype=float)
  else:
    diff = np.array(x, dtype=float)
  diff = diff[~np.isnan(diff)]
  diff = diff[diff != 0]
  if len(diff) == 0:
    return np.nan, np.nan
  res = stats.wilcoxon(diff)
  return res.statistic, res.pvalue


def safe_mannwhitneyu(x, y):
  x, y = np.array(x, dtype=float), np.array(y, dtype=float)
  x = x[~np.isnan(x)]
  y = y[~np.isnan(y)]
  if len(x) == 0 or len(y) == 0:
    return np.nan, np.nan
  res = stats.mannwhitneyu(x, y)
  return res.statistic, res.pvalue


def safe_ttest_ind(x, y):
  x = np.array(x, dtype=float)
  y = np.array(y, dtype=float)
  x = x[~np.isnan(x)]
  y = y[~np.isnan(y)]
  if len(x) < 2 or len(y) < 2:
    return np.nan, np.nan
  res = stats.ttest_ind(x, y)
  return res.statistic, res.pvalue


def safe_ttest_rel(x, y):
  """成對樣本 t 檢定 (Paired t-test)"""
  x = np.array(x, dtype=float)
  y = np.array(y, dtype=float)
  valid = (~np.isnan(x)) & (~np.isnan(y))
  x = x[valid]
  y = y[valid]
  if len(x) < 2:
    return np.nan, np.nan
  res = stats.ttest_rel(x, y)
  return res.statistic, res.pvalue


def safe_ttest_1samp(x, popmean=0):
  """單樣本 t 檢定 (One-sample t-test against popmean)"""
  x = np.array(x, dtype=float)
  x = x[~np.isnan(x)]
  if len(x) < 2:
    return np.nan, np.nan
  res = stats.ttest_1samp(x, popmean)
  return res.statistic, res.pvalue


# ==========================================
# LMM 與 Mixed ANOVA 分析函式
# ==========================================
def run_lmm_analysis():
  """將字典轉換為 DataFrame 並執行 LMM 分析"""
  try:
    import pandas as pd
    import statsmodels.formula.api as smf
  except ImportError:
    print(
        '\n[提示] 系統缺少 pandas 或 statsmodels 套件，無法執行 LMM'
        ' 分析。請使用 pip install pandas statsmodels 安裝。'
    )
    return

  rows = []
  for sub_int, data in raw_data_13.items():
    sub = str(sub_int)
    seq = 'Seq_AB' if data['s1']['cond'] == 'A' else 'Seq_BA'

    for sess in ['s1', 's2']:
      cond = data[sess]['cond']
      acc = data[sess]['off_sess'] * 100

      spectral_raw = [
          float(x) for x in data[sess].get('Spectral Saliency', []) if x != '...'
      ]
      spatial_raw = [
          float(x) for x in data[sess].get('Spatial Saliency', []) if x != '...'
      ]
      spectral = np.mean(spectral_raw) if spectral_raw else np.nan
      spatial = np.mean(spatial_raw) if spatial_raw else np.nan

      rows.append({
          'Subject': sub,
          'Session': sess.upper(),
          'Condition': cond,
          'Sequence': seq,
          'Accuracy': acc,
          'Spectral Saliency': spectral,
          'Spatial Saliency': spatial,
      })

  df = pd.DataFrame(rows)

  print('\n' + '=' * 130)
  print('📊 Table 3: Linear Mixed Model (LMM) & Two-way Mixed ANOVA')
  print('   * 模型公式: Metric ~ Session * Sequence + (1|Subject)')
  print('   * [Session 主效應] = 學習效應 (Learning Effect, LE)')
  print('   * [Sequence 主效應] = 順序/組別效應 (Carryover Effect)')
  print(
      '   * [Session × Sequence 交互作用] = 系統效應 (Adaptive Effect, AE)'
      ' - 等價於 Condition 主效應'
  )
  print('=' * 130)
  print(
      f"| {'Metric':<18} | {'Session (LE) p-val':<20} | {'Sequence p-val':<20} |"
      f" {'Interaction (AE) p-val':<24} |"
  )
  print(f"|{'-' * 20}|{'-' * 22}|{'-' * 22}|{'-' * 26}|")

  for m in ['Accuracy', 'Spectral Saliency', 'Spatial Saliency']:
    df_clean = df.dropna(subset=[m])
    if len(df_clean) == 0:
      continue

    try:
      formula = f'Q("{m}") ~ C(Session) * C(Sequence)'
      md = smf.mixedlm(formula, df_clean, groups=df_clean['Subject'])
      mdf = md.fit(disp=False)

      p_sess = mdf.pvalues.get('C(Session)[T.S2]', np.nan)
      seq_key = [
          k for k in mdf.pvalues.keys() if 'C(Sequence)' in k and ':' not in k
      ]
      p_seq = mdf.pvalues.get(seq_key[0], np.nan) if seq_key else np.nan
      int_key = [k for k in mdf.pvalues.keys() if ':' in k]
      p_int = mdf.pvalues.get(int_key[0], np.nan) if int_key else np.nan

      def fmt_p(p):
        if pd.isna(p):
          return 'N/A'
        return f'{p:.4f}' + ('*' if p < 0.05 else '')

      print(
          f'| {m:<18} | {fmt_p(p_sess):<20} | {fmt_p(p_seq):<20} |'
          f' {fmt_p(p_int):<24} |'
      )

    except Exception as e:
      print(f'| {m:<18} | 模型擬合失敗: {str(e)[:40]:<68} |')

  print('=' * 130)


# ==========================================
# 主要繪表邏輯
# ==========================================
def calculate_and_print_tables():
  ids = sorted([sub_int for sub_int in raw_data_13.keys()])
  metrics = ['Accuracy', 'Spectral Saliency', 'Spatial Saliency']

  g1_deltas = {m: [] for m in metrics}
  g2_deltas = {m: [] for m in metrics}

  paired_data = {
      m: {'s1': [], 's2': [], 'static': [], 'adaptive': []} for m in metrics
  }

  for sub in ids:
    data = raw_data_13[sub]

    s1_acc = data['s1']['off_sess'] * 100
    s2_acc = data['s2']['off_sess'] * 100

    s1_spectral_raw = [
        float(x) for x in data['s1'].get('Spectral Saliency', []) if x != '...'
    ]
    s2_spectral_raw = [
        float(x) for x in data['s2'].get('Spectral Saliency', []) if x != '...'
    ]
    s1_spectral = np.mean(s1_spectral_raw) if s1_spectral_raw else np.nan
    s2_spectral = np.mean(s2_spectral_raw) if s2_spectral_raw else np.nan

    s1_spatial_raw = [
        float(x) for x in data['s1'].get('Spatial Saliency', []) if x != '...'
    ]
    s2_spatial_raw = [
        float(x) for x in data['s2'].get('Spatial Saliency', []) if x != '...'
    ]
    s1_spatial = np.mean(s1_spatial_raw) if s1_spatial_raw else np.nan
    s2_spatial = np.mean(s2_spatial_raw) if s2_spatial_raw else np.nan

    vals = {
        'Accuracy': (s1_acc, s2_acc),
        'Spectral Saliency': (s1_spectral, s2_spectral),
        'Spatial Saliency': (s1_spatial, s2_spatial),
    }

    cond_s1 = data['s1']['cond']

    for m in metrics:
      v_s1, v_s2 = vals[m]
      delta = v_s2 - v_s1

      paired_data[m]['s1'].append(v_s1)
      paired_data[m]['s2'].append(v_s2)

      if cond_s1 == 'A':  # Group 1 (Seq A: A -> B)
        g1_deltas[m].append(delta)
        paired_data[m]['static'].append(v_s1)
        paired_data[m]['adaptive'].append(v_s2)
      else:  # Group 2 (Seq B: B -> A)
        g2_deltas[m].append(delta)
        paired_data[m]['adaptive'].append(v_s1)
        paired_data[m]['static'].append(v_s2)

  # ==========================================
  # Table 1: Sequence Effect
  # ==========================================
  print('\n' + '=' * 130)
  print(
      '📊 Table 1: Between-group comparison (Group 1 vs Group 2) for Sequence'
      ' Effect'
  )
  print('   * W-Test = Mann-Whitney U test (Wilcoxon rank-sum test)')
  print('   * t-Test = Independent Two-Sample t-test')
  print('=' * 130)
  print(
      f"| {'Metric':<18} | {'Group 1 Mean ± SD':<18} | {'Group 2 Mean ± SD':<18}"
      f" | {'Mean Diff':<12} | {'U-Stat':<8} | {'W-p-val':<8} | {'t-Stat':<8} |"
      f" {'t-p-val':<8} |"
  )
  print(
      f"|{'-' * 20}|{'-' * 20}|{'-' * 20}|{'-' * 14}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|"
  )

  for m in metrics:
    v1 = np.array(g1_deltas[m])
    v2 = np.array(g2_deltas[m])

    v1 = v1[~np.isnan(v1)]
    v2 = v2[~np.isnan(v2)]

    m1, std1 = np.mean(v1), np.std(v1, ddof=1)
    m2, std2 = np.mean(v2), np.std(v2, ddof=1)
    diff = m1 - m2

    w_stat, w_p = safe_mannwhitneyu(v1, v2)
    t_stat, t_p = safe_ttest_ind(v1, v2)

    g1_str = f'{m1:.2f} ± {std1:.2f}'
    g2_str = f'{m2:.2f} ± {std2:.2f}'

    print(
        f'| {m:<18} | {g1_str:>18} | {g2_str:>18} | {diff:>12.2f} |'
        f' {w_stat:>8.2f} | {w_p:>8.4f} | {t_stat:>8.2f} | {t_p:>8.4f} |'
    )
  print('=' * 130)

  # ==========================================
  # Table 2: Paired Samples Tests
  # ==========================================
  print('\n' + '=' * 140)
  print(
      '📊 Table 2: Paired-Samples Tests for Learning Effect (LE) and Adaptive'
      ' Effect (AE)'
  )
  print('   * LE Paired Test: Session 2 vs Session 1 (成對樣本比較)')
  print(
      '   * AE Paired Test: Adaptive (Cond B) vs Static (Cond A) (成對樣本比較)'
  )
  print(
      '   * Pair-t: Paired t-test (母數) | W-Stat / W-p: Wilcoxon Signed-Rank'
      ' Test (無母數)'
  )
  print('=' * 140)
  print(
      f"| {'Metric':<18} | {'LE Diff':<8} | {'LE W-Stat':<9} | {'LE W-p':<8} |"
      f" {'LE Pair-t':<10} | {'LE t-p':<8} | {'AE Diff':<8} | {'AE W-Stat':<9}"
      f" | {'AE W-p':<8} | {'AE Pair-t':<10} | {'AE t-p':<8} |"
  )
  print(
      f"|{'-' * 20}|{'-' * 10}|{'-' * 11}|{'-' * 10}|{'-' * 12}|{'-' * 10}|{'-' * 10}|{'-' * 11}|{'-' * 10}|{'-' * 12}|{'-' * 10}|"
  )

  for m in metrics:
    s1_arr = np.array(paired_data[m]['s1'])
    s2_arr = np.array(paired_data[m]['s2'])
    valid_le = (~np.isnan(s1_arr)) & (~np.isnan(s2_arr))
    s1_clean = s1_arr[valid_le]
    s2_clean = s2_arr[valid_le]

    le_diff = np.mean(s2_clean - s1_clean)
    le_w_stat, le_w_p = safe_wilcoxon(s2_clean, s1_clean)
    le_t_stat, le_t_p = safe_ttest_rel(s2_clean, s1_clean)

    ad_arr = np.array(paired_data[m]['adaptive'])
    st_arr = np.array(paired_data[m]['static'])
    valid_ae = (~np.isnan(ad_arr)) & (~np.isnan(st_arr))
    ad_clean = ad_arr[valid_ae]
    st_clean = st_arr[valid_ae]

    ae_diff = np.mean(ad_clean - st_clean)
    ae_w_stat, ae_w_p = safe_wilcoxon(ad_clean, st_clean)
    ae_t_stat, ae_t_p = safe_ttest_rel(ad_clean, st_clean)

    print(
        f'| {m:<18} | {le_diff:>8.2f} | {le_w_stat:>9.2f} | {le_w_p:>8.4f} |'
        f' {le_t_stat:>10.2f} | {le_t_p:>8.4f} | {ae_diff:>8.2f} |'
        f' {ae_w_stat:>9.2f} | {ae_w_p:>8.4f} | {ae_t_stat:>10.2f} |'
        f' {ae_t_p:>8.4f} |'
    )
  print('=' * 140)

  # ==========================================
  # Table 2b: One-Sample Tests on Deltas
  # ==========================================
  print('\n' + '=' * 140)
  print('📊 Table 2b: One-Sample Tests on Differences against 0 (Δ vs 0 檢定)')
  print('   * LE Δ = Session 2 - Session 1  |  AE Δ = Adaptive (Cond B) - Static (Cond A)')
  print('   * 1Samp-t: One-Sample t-test against 0 (測試平均差值是否顯著異於 0)')
  print('   * W-Stat / W-p: Wilcoxon Signed-Rank Test against 0 (無母數差值檢定)')
  print('=' * 140)
  print(
      f"| {'Metric':<18} | {'LE Mean Δ':<9} | {'LE W-Stat':<9} | {'LE W-p':<8} |"
      f" {'LE 1Samp-t':<10} | {'LE t-p':<8} | {'AE Mean Δ':<9} |"
      f" {'AE W-Stat':<9} | {'AE W-p':<8} | {'AE 1Samp-t':<10} | {'AE t-p':<8} |"
  )
  print(
      f"|{'-' * 20}|{'-' * 11}|{'-' * 11}|{'-' * 10}|{'-' * 12}|{'-' * 10}|{'-' * 11}|{'-' * 11}|{'-' * 10}|{'-' * 12}|{'-' * 10}|"
  )
  
  for m in metrics:
      s1_arr = np.array(paired_data[m]['s1'])
      s2_arr = np.array(paired_data[m]['s2'])
      valid_le = (~np.isnan(s1_arr)) & (~np.isnan(s2_arr))
      le_delta = s2_arr[valid_le] - s1_arr[valid_le]
  
      le_mean_d = np.mean(le_delta)
      le_w_stat, le_w_p = safe_wilcoxon(le_delta)
      le_1t_stat, le_1t_p = safe_ttest_1samp(le_delta, popmean=0)
  
      ad_arr = np.array(paired_data[m]['adaptive'])
      st_arr = np.array(paired_data[m]['static'])
      valid_ae = (~np.isnan(ad_arr)) & (~np.isnan(st_arr))
      ae_delta = ad_arr[valid_ae] - st_arr[valid_ae]
  
      ae_mean_d = np.mean(ae_delta)
      ae_w_stat, ae_w_p = safe_wilcoxon(ae_delta)
      ae_1t_stat, ae_1t_p = safe_ttest_1samp(ae_delta, popmean=0)
  
      # 修正：將末端的 {ae_t_p:>8.4f} 改為 {ae_1t_p:>8.4f}
      print(
          f'| {m:<18} | {le_mean_d:>9.2f} | {le_w_stat:>9.2f} | {le_w_p:>8.4f} |'
          f' {le_1t_stat:>10.2f} | {le_1t_p:>8.4f} | {ae_mean_d:>9.2f} |'
          f' {ae_w_stat:>9.2f} | {ae_w_p:>8.4f} | {ae_1t_stat:>10.2f} |'
          f' {ae_1t_p:>8.4f} |'
      )
  print('=' * 140)

  # ==========================================
  # Table 3: LMM 模型分析
  # ==========================================
  run_lmm_analysis()


if __name__ == '__main__':
  calculate_and_print_tables()