"""Generate the dependency-free architecture overview used in this report."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

WIDTH, HEIGHT = 1200, 680
BG = (248, 250, 252)
pixels = [list(BG) for _ in range(WIDTH * HEIGHT)]


def put(x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        pixels[y * WIDTH + x] = list(color)


def rect(x0: int, y0: int, x1: int, y1: int, fill, border=(30, 41, 59), width=3) -> None:
    for y in range(y0, y1):
        for x in range(x0, x1):
            put(x, y, border if x < x0 + width or x >= x1 - width or y < y0 + width or y >= y1 - width else fill)


def line(x0: int, y0: int, x1: int, y1: int, color=(71, 85, 105), width=3) -> None:
    dx, dy = x1-x0, y1-y0
    steps = max(abs(dx), abs(dy), 1)
    for i in range(steps + 1):
        x = round(x0 + dx*i/steps); y = round(y0 + dy*i/steps)
        for oy in range(-width//2, width//2+1):
            for ox in range(-width//2, width//2+1): put(x+ox, y+oy, color)


def circle(cx: int, cy: int, radius: int, fill, border=(30, 41, 59)) -> None:
    for y in range(cy-radius-2, cy+radius+3):
        for x in range(cx-radius-2, cx+radius+3):
            d=(x-cx)**2+(y-cy)**2
            if d <= radius**2: put(x,y, border if d >= (radius-3)**2 else fill)


def arrow(x0: int, y0: int, x1: int, y1: int, color=(15, 118, 110)) -> None:
    line(x0,y0,x1,y1,color,5); line(x1,y1,x1-14,y1-10,color,5); line(x1,y1,x1-14,y1+10,color,5)


# Seven pipeline panels. Their meanings are documented in reports/README.md.
colors=[(219,234,254),(224,231,255),(237,233,254),(243,232,255),(204,251,241),(220,252,231),(254,240,138)]
boxes=[(30,60,170,160),(205,60,345,160),(380,60,520,160),(555,60,695,160),(730,60,870,160),(905,25,1170,100),(905,120,1170,195)]
for box,color in zip(boxes,colors): rect(*box,color)
for a,b in zip(boxes[:4],boxes[1:5]): arrow(a[2],110,b[0],110)
arrow(870,110,905,65); arrow(870,110,905,158)

# Hierarchical electrode/rhythm graph: five colored rhythm nodes per region.
band_colors=[(96,165,250),(45,212,191),(250,204,21),(251,146,60),(244,63,94)]
centers=[]
for x in (150,370,590,810,1030):
    ys=(360,410,460,510,560); region_nodes=[]
    for y,color in zip(ys,band_colors): circle(x,y,14,color); region_nodes.append((x,y))
    centers.append(region_nodes)
    # local cross-frequency links
    for i in range(4): line(x,ys[i]+15,x,ys[i+1]-15,(148,163,184),2)
# same-band spatial links
for left,right in zip(centers,centers[1:]):
    for u,v,color in zip(left,right,band_colors): line(u[0]+15,u[1],v[0]-15,v[1],color,2)
# region pooling nodes and graph pooling node
for x in (150,370,590,810,1030): arrow(x,345,x,300,(100,116,139)); circle(x,285,18,(167,139,250))
for x in (150,370,590,810,1030): line(x,265,590,235,(139,92,246),3)
circle(590,220,23,(126,34,206))

# Legend swatches, intentionally text-free so generation has no font dependency.
for i,color in enumerate(band_colors): rect(90+i*105,625,145+i*105,650,color,width=2)
rect(700,625,755,650,(167,139,250),width=2); rect(870,625,925,650,(126,34,206),width=2)

def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)


def main() -> None:
    """Encode the deterministic in-memory drawing as an RGB PNG."""
    raw = b''.join(
        b'\x00' + bytes(channel for px in pixels[y*WIDTH:(y+1)*WIDTH] for channel in px)
        for y in range(HEIGHT)
    )
    png = (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw, 9))
        + chunk(b'IEND', b'')
    )
    output = Path(__file__).with_name('architecture_overview.png')
    output.write_bytes(png)
    print(f'wrote {output.name} ({WIDTH}x{HEIGHT})')


if __name__ == '__main__':
    main()
