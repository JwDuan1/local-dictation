"""One-time: draw dictate.ico (a white microphone on a red circle). Run with any Python that has Pillow."""
from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.ellipse((8, 8, S - 8, S - 8), fill=(198, 40, 40, 255))
white = (255, 255, 255, 255)
d.rounded_rectangle((98, 48, 158, 150), radius=30, fill=white)            # capsule
d.arc((74, 92, 182, 182), start=0, end=180, fill=white, width=14)         # cradle
d.line((128, 182, 128, 206), fill=white, width=14)                         # stem
d.line((98, 208, 158, 208), fill=white, width=14)                          # base
img.save(r"D:\LLM\dictate\dictate.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("saved")
