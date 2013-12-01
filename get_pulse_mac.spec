# -*- mode: python -*-
a = Analysis(['get_pulse.py'],
             pathex=[
                 '/Users/tristanhearn/Documents/thearn_repos/webcam-pulse-detector'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
a.datas += [
    ('haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt.xml', 'DATA')]
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='get_pulse',
          debug=False,
          strip=None,
          upx=True,
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='get_pulse')
app = BUNDLE(coll,
             name='get_pulse.app',
             icon=None)
