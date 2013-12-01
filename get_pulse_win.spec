# -*- mode: python -*-
a = Analysis(['get_pulse.py'],
             pathex=['D:\\webcam-pulse-detector'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
a.datas += [
    ('haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt.xml', 'DATA')]
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='get_pulse.exe',
          debug=True,
          strip=None,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='get_pulse')
