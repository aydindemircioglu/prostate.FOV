
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
from PIL import Image


fontSize = 64
fontColor = (255, 255, 255)
fontFace = "Arial"
fontPosX = -64
fontPosY = -50


def addText (finalImage, text = '', org = (0,0), fontFace = 'Arial', fontSize = 12, color = (255,255,255)):
     draw = ImageDraw.Draw(finalImage)
     font = ImageFont.truetype(fontFace + ".ttf", fontSize)
     draw.text(org, text, font=font, fill = color)
     return (finalImage.copy())



paths = ['slices/2731904.png', 'slices/2985174.png', 'slices/2260247.png']
imgs = [Image.open(p) for p in paths]
target_size = (imgs[0].size[0] * 2, imgs[0].size[1] * 2)
processed_imgs = [img.resize(target_size) for img in imgs]

canvas_w = (target_size[0] * 3) + 120
canvas_h = target_size[1] + 60
canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
draw = ImageDraw.Draw(canvas)

arrow_data = [
    [
        {'tip': (273, 320), 'len': 80, 'dir': 'V'},
        {'tip': (300, 390), 'len': 80, 'dir': '-H'},
        {'tip': (225, 390), 'len': 80, 'dir': 'H'}
    ],
    [
        {'tip': (223, 175), 'len': 80, 'dir': 'V'},
        {'tip': (184, 346), 'len': 80, 'dir': 'H'},
        {'tip': (335, 346), 'len': 80, 'dir': '-H'}
    ],
    [
        {'tip': (295-15, 404), 'len': 80, 'dir': 'H'},
        {'tip': (295+15, 404), 'len': 80, 'dir': '-H'}
    ]
]

for i, img in enumerate(processed_imgs):
    offset_x = 30 + i * (target_size[0] + 30)
    offset_y = 30
    addText(img, text=chr(ord('A')+i), org = (15,590), fontSize = 40, color = (255,255,255))
    canvas.paste(img, (offset_x, offset_y))

    for a in arrow_data[i]:
        tx, ty = a['tip'][0] + offset_x, a['tip'][1] + offset_y
        length = a['len']
        d = a['dir']

        if d == 'V':
            base = (tx, ty - length)
            head = [(tx - 5, ty - 8), (tx + 5, ty - 8), (tx, ty)]
        elif d == '-V':
            base = (tx, ty + length)
            head = [(tx - 5, ty + 8), (tx + 5, ty + 8), (tx, ty)]
        elif d == 'H':
            base = (tx - length, ty)
            head = [(tx - 8, ty - 5), (tx - 8, ty + 5), (tx, ty)]
        elif d == '-H':
            base = (tx + length, ty)
            head = [(tx + 8, ty - 5), (tx + 8, ty + 5), (tx, ty)]

        draw.line([base, (tx, ty)], fill='red', width=2)
        draw.polygon(head, fill='red')

canvas.save('./Figure_Slices.tiff')
