from torchvision import transforms
import torch
import numpy as np
from pathlib import Path

class RescaleTransform(object):
  """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, the greater image
            dimension is matched to one of the output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
  def __init__(self, output_size):
      assert isinstance(output_size, (int, tuple))
      self.output_size = output_size

  def __call__(self, image):
        h, w = image.shape[1:]
        aspect_ratio = w / h
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
            if w > h:
                new_h = int(new_w / aspect_ratio)
            else:
                new_w = int(new_h * aspect_ratio)
        else:
            new_h, new_w = self.output_size
            if h > w:
                new_w = int(new_h * aspect_ratio)
            else:
                new_h = int(new_w / aspect_ratio)

        new_h, new_w = int(new_h), int(new_w)
        image = transforms.Resize((new_h, new_w), antialias=True)(image)
        return image

class PadTransform(object):
    """Pad the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, the pad will be a square.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):
        if isinstance(self.output_size, int):
          background = torch.zeros((3, self.output_size, self.output_size))
          background[:,0:img.shape[1],0:img.shape[2]] += img
        else:
          channels, height, width = 3, self.output_size[0], self.output_size[1]
          background = torch.zeros((channels, height, width))
          background[:, 0:img.shape[1], 0:img.shape[2]] += img
        return background