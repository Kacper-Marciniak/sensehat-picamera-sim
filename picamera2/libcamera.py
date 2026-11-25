"""Tiny stand-in for ``libcamera``: only ``Transform`` (image flips)."""
 
 
class Transform:
    def __init__(self, hflip=0, vflip=0, transpose=0):
        self.hflip, self.vflip, self.transpose = bool(hflip), bool(vflip), bool(transpose)
 
    def __repr__(self):
        return f"Transform(hflip={int(self.hflip)}, vflip={int(self.vflip)}, transpose={int(self.transpose)})"