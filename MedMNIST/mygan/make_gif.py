# -*- coding: utf-8 -*-
import imageio
import os
import re
import numpy as np

def compose_gif(path_, img_paths):
    
    gif_images = []
    for path in img_paths:
        gif_images.append(imageio.imread(path))
    imageio.mimsave(path_+"/test.gif",gif_images,duration=1)

for i in ['gan', 'wgan']:
    for j in ['0', '1']:
        path_ = os.path.join('./images', '%s/%s' %(i, j))
        a = [os.path.join(path_, i) for i in os.listdir(path_) if i.startswith('epoch')]
        indexes = [int(re.search(r'epoch_(\d+)', i).group(1)) for i in a]
        s = np.array(a)[np.argsort(indexes)]
        assert len(s) == 200, '长度不匹配%s, %s' % (i, j)
        compose_gif(path_, s)