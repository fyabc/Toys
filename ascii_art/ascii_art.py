#! /usr/bin/python
# -*- coding: utf-8 -*-

"""ASCII art generator."""

import argparse
import os
from bisect import bisect
from pathlib import Path

from PIL import ImageFont, Image, ImageDraw

__author__ = 'fyabc'

SYS_FONT_DIR = Path('C:/Windows/Fonts')
FONT = ImageFont.load_default()
N_COLORS = 256
CHARSET = [chr(i) for i in range(32, 127)]


def get_charset_grey():
    def get_grey(char):
        """Get the gray scale of the char."""

        size = FONT.getsize(char)
        image = Image.new('1', size)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), char, fill='white', font=FONT)

        n_white = list(image.getdata()).count(N_COLORS - 1)
        return n_white / (size[0] * size[1])

    charset_grey = sorted([(c, get_grey(c)) for c in CHARSET], key=lambda e: e[1])

    # Scale by the max grey scale value
    max_grey = charset_grey[-1][1]
    charset_grey = [(c, g / max_grey * (N_COLORS - 1)) for c, g in charset_grey]

    return charset_grey


CHARSET_GREY = get_charset_grey()


def set_font(args):
    global FONT
    if args.font is None:
        FONT = ImageFont.load_default()
    else:
        FONT = ImageFont.truetype(str(SYS_FONT_DIR / (args.font + '.ttf')), size=args.fontsize)


def histogram(image):
    """Count colors in the image."""

    result = [0] * N_COLORS

    for i in image:
        result[i] += 1

    return result


def transform_color(color_histogram):
    """Transform colors of the histogram into new colors."""

    total = sum(color_histogram)
    result = [0] * N_COLORS

    s = 0
    for i, n in enumerate(color_histogram):
        s += n
        result[i] = int((N_COLORS - 1) * s / total)

    return result


def transform_image(image_data, mapping):
    """Transform pixels in image into new value by mapping."""

    return [mapping[e] for e in image_data]


def search_nearest_grey(g, charset_grey: list):
    """Search nearest grey scale of g in the charset gray. Using binary search."""

    i = bisect([g for c, g in charset_grey], g)
    if i == len(charset_grey):
        return charset_grey[i - 1][0]
    if abs(charset_grey[i - 1][1] - g) < abs(charset_grey[i][1] - g):
        return charset_grey[i - 1][0]
    else:
        return charset_grey[i][0]


def get_image_string(image_data: list, charset_grey):
    """Get image string from image data."""

    return ''.join(search_nearest_grey(g, charset_grey) for g in image_data)


def ascii_art(image_path: str, char_width=80):
    """Draw the ASCII art from the image."""

    orig_image = Image.open(image_path)
    is_animated = False

    if getattr(orig_image, 'is_animated', False):
        is_animated = True
        image_list = []
        for i in range(orig_image.n_frames):
            orig_image.seek(i)
            image_list.append(orig_image.convert('L'))
    else:
        image_list = [orig_image.convert('L')]

    image_string_list = []
    for image in image_list:
        char_height = int(char_width / image.size[0] * image.size[1])
        image = image.resize((char_width, char_height))
        image_data = list(image.getdata())

        new_image_data = transform_image(image_data, transform_color(histogram(image_data)))

        image_string = get_image_string(new_image_data, CHARSET_GREY)
        image_string = '\n'.join(image_string[i * char_width: (i + 1) * char_width] for i in range(char_height))
        image_string_list.append(image_string)

    print(f'| Parsed image into strings (size={image_list[0].size}, {len(image_list)} frames)')

    return image_string_list, is_animated


def ascii_art_to_image(image_string: str, echo=True):
    """Draw the ASCII art into an image."""

    lines = image_string.split('\n')
    s_width, s_height = len(lines[0]), len(lines)

    char_size = FONT.getsize('a')
    char_size = (char_size[0] + 2, char_size[1] + 2)

    image = Image.new('1', (char_size[0] * s_width, char_size[1] * s_height))
    draw = ImageDraw.Draw(image)

    for i, line in enumerate(lines):
        for j, c in enumerate(line):
            draw.text((j * char_size[0], i * char_size[1]), c, fill='white', font=FONT)
        if echo:
            if (i + 1) % 10 == 0:
                print('.', end='', flush=True)

    return image


def main():
    parser = argparse.ArgumentParser('Generate ASCII art from given image.')
    parser.add_argument('input', default=None, help='Input image file')
    parser.add_argument('output', default=None, nargs='?', help='Output ASCII art image file, default is None')
    parser.add_argument('-f', '--font', action='store', default=None, help='Font to use, default is None')
    parser.add_argument('--fs', '--font-size', action='store', dest='fontsize', default=12, type=int,
                        help='Font size, default is %(default)s')
    parser.add_argument('-s', '--size', action='store', dest='size', default=80, type=int,
                        help='Width of output file, default is %(default)s')

    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), 'images', 'ascii-' + os.path.basename(args.input))

    set_font(args)
    image_string_list, is_animated = ascii_art(args.input, char_width=args.size)

    new_image_list = []
    for i, image_string in enumerate(image_string_list):
        print(f'Frame {i} ', end='', flush=True)
        new_image_list.append(ascii_art_to_image(image_string, echo=True))
        print(flush=True)

    if is_animated:
        frame0 = new_image_list[0].convert('P')
        frame0.save(args.output, save_all=True, append_images=[frame.convert('P') for frame in new_image_list[1:]])
    else:
        assert len(new_image_list) == 1
        new_image = new_image_list[0]
        new_image.save(args.output)
    # print(s)


if __name__ == '__main__':
    main()
