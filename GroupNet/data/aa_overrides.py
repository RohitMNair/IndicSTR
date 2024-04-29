# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extends default ops to accept optional parameters."""
from functools import partial

from timm.data.auto_augment import _LEVEL_DENOM, LEVEL_TO_ARG, NAME_TO_OP, _randomly_negate, rotate, _check_args_tf
from PIL import Image

def rotate_expand(img, degrees, **kwargs):
    """Rotate operation with expand=True to avoid cutting off the characters"""
    kwargs['expand'] = True
    return rotate(img, degrees, **kwargs)

def shear_x_expand(img, factor, **kwargs):
    _check_args_tf(kwargs)
    # Calculate the amount of padding needed
    pad_x = int(abs(factor) * img.size[1])
    # Create a new image with padding
    padded_width = img.size[0] + pad_x
    padded_img = Image.new(img.mode, (padded_width, img.size[1]), kwargs.get('fillcolor'))
    # Paste the original image onto the padded image
    if factor >= 0:
        padded_img.paste(img, (pad_x, 0))
    else:
        padded_img.paste(img, (0, 0))
    # Apply shear transformation
    sheared_img = padded_img.transform(padded_img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)
    # Resize back to original dimensions
    # sheared_img = sheared_img.resize(img.size, Image.ANTIALIAS)
    return sheared_img

def shear_y_expand(img, factor, **kwargs):
    _check_args_tf(kwargs)
    # Calculate the amount of padding needed
    pad_y = int(abs(factor) * img.size[0])
    # Create a new image with padding
    padded_height = img.size[1] + pad_y
    padded_img = Image.new(img.mode, (img.size[0], padded_height), kwargs.get('fillcolor'))
    # Paste the original image onto the padded image
    if factor >= 0:
        padded_img.paste(img, (0, pad_y))
    else:
        padded_img.paste(img, (0, 0))
    # Apply shear transformation
    sheared_img = padded_img.transform(padded_img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)
    # Resize back to original dimensions
    # sheared_img = sheared_img.resize(img.size, Image.ANTIALIAS)
    return sheared_img

def translate_x_rel_expand(img, pct, **kwargs):
    pixels = int(pct * img.size[0])
    _check_args_tf(kwargs)
    # Calculate the amount of padding needed
    pad_x = abs(pixels)
    # Create a new image with padding
    padded_img = Image.new(img.mode, (img.size[0] + pad_x, img.size[1]), kwargs.get('fillcolor'))
    # Paste the original image onto the padded image
    if pixels > 0:
        padded_img.paste(img, (pad_x, 0))
    else:
        padded_img.paste(img, (0, 0))
    # Apply translation transformation
    translated_img = padded_img.transform(padded_img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)
    # Resize back to original dimensions
    # translated_img = translated_img.resize(img.size, Image.ANTIALIAS)
    return translated_img

def translate_y_rel_expand(img, pct, **kwargs):
    pixels = int(pct * img.size[1])
    _check_args_tf(kwargs)
    # Calculate the amount of padding needed
    pad_y = abs(pixels)
    # Create a new image with padding
    padded_img = Image.new(img.mode, (img.size[0], img.size[1] + pad_y), kwargs.get('fillcolor'))
    # Paste the original image onto the padded image
    if pixels > 0:
        padded_img.paste(img, (0, pad_y))
    else:
        padded_img.paste(img, (0, 0))
    # Apply translation transformation
    translated_img = padded_img.transform(padded_img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)
    # Resize back to original dimensions
    # translated_img = translated_img.resize(img.size, Image.ANTIALIAS)
    return translated_img

def translate_x_abs_expand(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    # Calculate the amount of padding needed
    pad_x = abs(pixels)
    # Create a new image with padding
    padded_img = Image.new(img.mode, (img.size[0] + pad_x, img.size[1]), kwargs.get('fillcolor'))
    # Paste the original image onto the padded image
    if pixels > 0:
        padded_img.paste(img, (pad_x, 0))
    else:
        padded_img.paste(img, (0, 0))
    # Apply translation transformation
    translated_img = padded_img.transform(padded_img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)
    # Resize back to original dimensions
    translated_img = translated_img.resize(img.size, Image.ANTIALIAS)
    return translated_img

def translate_y_abs_expand(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    # Calculate the amount of padding needed
    pad_y = abs(pixels)
    # Create a new image with padding
    padded_img = Image.new(img.mode, (img.size[0], img.size[1] + pad_y), kwargs.get('fillcolor'))
    # Paste the original image onto the padded image
    if pixels > 0:
        padded_img.paste(img, (0, pad_y))
    else:
        padded_img.paste(img, (0, 0))
    # Apply translation transformation
    translated_img = padded_img.transform(padded_img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)
    # Resize back to original dimensions
    translated_img = translated_img.resize(img.size, Image.ANTIALIAS)
    return translated_img

def _level_to_arg(level, hparams, key, default):
    magnitude = hparams.get(key, default)
    level = (level / _LEVEL_DENOM) * magnitude
    level = _randomly_negate(level)
    return (level,)

def apply():
    # Overrides
    NAME_TO_OP.update({
        'Rotate': rotate_expand,
        'ShearX': shear_x_expand,
        'ShearY': shear_y_expand,
        'TranslateXRel': translate_x_rel_expand,
        'TranslateYRel': translate_y_rel_expand,
        'TranslateX': translate_x_abs_expand,
        'TranslateY': translate_y_abs_expand,
    })
    LEVEL_TO_ARG.update({
        'Rotate': partial(_level_to_arg, key='rotate_deg', default=30.0),
        'ShearX': partial(_level_to_arg, key='shear_x_pct', default=0.3),
        'ShearY': partial(_level_to_arg, key='shear_y_pct', default=0.3),
        'TranslateXRel': partial(_level_to_arg, key='translate_x_pct', default=0.45),
        'TranslateYRel': partial(_level_to_arg, key='translate_y_pct', default=0.45),
    })