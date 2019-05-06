# 22 features used (of the bosphorus dataset)
features = ['Outer left eyebrow', 'Middle left eyebrow', 'Inner left eyebrow', 'Inner right eyebrow', 'Middle right eyebrow', 'Outer right eyebrow',
            'Outer left eye corner', 'Inner left eye corner', 'Inner right eye corner', 'Outer right eye corner',
            'Nose saddle left', 'Nose saddle right', 'Left nose peak', 'Nose tip', 'Right nose peak', 'Left mouth corner',
            'Upper lip outer middle', 'Right mouth corner', 'Upper lip inner middle', 'Lower lip inner middle', 'Lower lip outer middle', 'Chin middle']

# maps the features (bosphorus dataset) to the index of the landmark in our test dataset,
# # annoation method for test dataset by Adrian Bulat: https://github.com/1adrianb/face-alignment
# # index: 0-67, None = the feature in the bosphorus dataset doesn't exist in our annotation method
feature_mapping = {
    'Chin middle': 8,
    'Inner left eye corner': 39,
    'Inner left eyebrow': 21,
    'Inner right eye corner': 42,
    'Inner right eyebrow': 22,
    'Left mouth corner': 48,
    'Left nose peak': None,
    'Lower lip inner middle': 66,
    'Lower lip outer middle': 57,
    'Middle left eyebrow': 19,
    'Middle right eyebrow': 24,
    'Nose saddle left': None,
    'Nose saddle right': None,
    'Nose tip': 30,
    'Outer left eye corner': 36,
    'Outer left eyebrow': 17,
    'Outer right eye corner': 45,
    'Outer right eyebrow': 26,
    'Right mouth corner': 54,
    'Right nose peak': None,
    'Upper lip inner middle': 62,
    'Upper lip outer middle': 51
}
