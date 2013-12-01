# -*- mode: python -*-
a = Analysis(['get_pulse.py'],
             pathex=['D:\\webcam-pulse-detector'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
a.datas += [
    ('haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt.xml', 'DATA')]
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='get_pulse.exe',
          debug=False,
          strip=None,
          upx=True,
          console=True )
